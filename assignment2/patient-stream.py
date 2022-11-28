import numpy as np # version 1.23.3
import simpy # version 4.0.1
# Python version 3.10.6
import random
from scipy.stats import t

# Constants of the simulation. Assuming that time is in minutes
N_PREPARATION_FACILITIES = 3
N_OPERATION_FACILITIES = 1
N_RECOVERY_FACILITIES = 3
PREPARATION_TIME = 40
OPERATION_TIME = 20
RECOVERY_TIME = 40
PATIENT_INTERARRIVAL_TIME = 24
SIM_TIME = 10000
COMPLICATION_CHANGE = 0.01 # If complications happen during the operation, patient will skip recovery and go to ER instead
RANDOM_SEED = 100 # Set to none if don't want to use seed
VERBOSE = 1

class Logger:
  verbose = 5
  def __init__(self, verbose=5):
    self.verbose = verbose

  def set_verbose(self, verbose):
    self.verbose = verbose

  def log(self, msg, verbose_level = 5):
    if verbose_level <= self.verbose:
      print(msg)
    else:
      return

class FacilitiesManager:
  """Managing resources of failicities processing their services"""
  preparation_facilities = None
  operation_facilities = None
  recovery_facilities = None
  prep_time = None
  op_time = None
  rec_time = None
  compl_ratio = None
  env = None
  logger = None
  include_twist = None

  def __init__(self, env, verbose=VERBOSE, p = N_PREPARATION_FACILITIES, o = N_OPERATION_FACILITIES, r = N_RECOVERY_FACILITIES, prep_time = PREPARATION_TIME, op_time = OPERATION_TIME, rec_time = RECOVERY_TIME, compl_chance = COMPLICATION_CHANGE, include_twist = True):
    self.logger = Logger(verbose=verbose)
    self.env = env
    self.preparation_facilities = simpy.Resource(env, p)
    self.operation_facilities = simpy.Resource(env, o)
    self.recovery_facilities = simpy.PriorityResource(env, r)
    self.prep_time = prep_time
    self.op_time = op_time
    self.rec_time = rec_time
    self.compl_ratio = compl_chance
    self.include_twist = include_twist

  def preparation(self, patient):
    self.logger.log(f"{patient} enters preparation facility at {self.env.now:.2f}", 5)
    patient.preparation_start_time = float(self.env.now)
    random_time = np.random.exponential(scale=self.prep_time)
    self.logger.log(f"{patient} preparation time is {random_time}")
    yield self.env.timeout(random_time)
    self.logger.log(f"  {patient} leaves preparation facility at {self.env.now:.2f}", 5)
    patient.preparation_end_time = float(self.env.now)

  def operation(self, patient):
    self.logger.log(f"{patient} enters operation facility at {self.env.now:.2f}", 5)
    patient.dataset.operation_block_time += float(self.env.now) - patient.preparation_end_time
    patient.operation_start_time = float(self.env.now)
    random_time = np.random.exponential(scale=self.op_time)
    self.logger.log(f"{patient} operation time is {random_time}")
    yield self.env.timeout(random_time)
    if (self.include_twist and np.random.uniform(0, 1) <= self.compl_ratio):
      self.logger.log(f"Complication occurred for {patient}, patient is put to high priority recovery queue", 5)
      patient.treatment_priority = 1
      patient.complication_occurred = True
      patient.dataset.patient_complications += 1
    else:
      patient.complication_occurred = False
    self.logger.log(f"  {patient} leaves operation facility at {self.env.now:.2f}", 5)
    patient.operation_end_time = float(self.env.now)
    patient.dataset.operation_time += patient.operation_end_time - patient.operation_start_time

  def recovery(self, patient):
    self.logger.log(f"{patient} enters recovery facility at {self.env.now:.2f}", 5)
    patient.recovery_start_time = float(self.env.now)
    random_time = np.random.exponential(scale=self.rec_time if patient.treatment_priority > 1 or not self.include_twist else self.rec_time*2)
    self.logger.log(f"{patient} recovery time is {random_time}")
    yield self.env.timeout(random_time)
    self.logger.log(f"  {patient} leaves recovery facility at {self.env.now:.2f}", 5)
    patient.recovery_end_time = float(self.env.now)
    patient.dataset.recovery_time += patient.recovery_end_time - patient.recovery_start_time
    patient.dataset.recovery_full_time += patient.recovery_start_time - patient.operation_end_time

class Patient:
  """Manages single patient's life cycle. Stores data of individual patient and keeps track of 
  all arrived patients and treated patients"""
  patients_num = 0
  treatment_priority = 10
  arrival_time = None
  preparation_queue_length = None # On entering queue
  operation_queue_length = None # On entering queue
  recovery_queue_length = None # On entering queue
  operation_block_time = None
  operation_time = None
  preparation_start_time = None
  preparation_end_time = None
  operation_start_time = None
  operation_end_time = None
  recovery_start_time = None
  recovery_end_time = None
  complication_occurred = None
  dataset = None
  facilities = None
  env = None
  logger = None

  def __init__(self, env, facilities, dataset, verbose=VERBOSE):
    self.treatment_priority = 10
    self.patients_num = 0
    self.arrival_time = None
    self.preparation_start_time = None
    self.preparation_end_time = None
    self.operation_start_time = None
    self.preparation_queue_length = None
    self.operation_queue_length = None
    self.recovery_queue_length = None
    self.operation_block_time = None
    self.operation_time = None
    self.operation_end_time = None
    self.recovery_start_time = None
    self.recovery_end_time = None
    self.complication_occurred = None
    self.dataset = dataset
    self.facilities = facilities
    self.logger = Logger(verbose)
    self.env = env
    self.dataset.patients_arrived += 1
    self.patients_num = self.dataset.patients_arrived
    self.logger.log(f"{self} arrives at {env.now:.2f}", 5)
    self.arrival_time = float(env.now)

  def arrival(self):
    prep_q = len(self.facilities.preparation_facilities.queue)
    prep_c = self.facilities.preparation_facilities.count
    self.preparation_queue_length = prep_q
    with self.facilities.preparation_facilities.request() as request:
      yield request
      yield self.env.process(self.facilities.preparation(self))
      self.dataset.patients_prepared += 1
      self.facilities.preparation_facilities.release(request)

    op_q = len(self.facilities.operation_facilities.queue)
    op_c = self.facilities.operation_facilities.count
    self.operation_queue_length = op_q
    with self.facilities.operation_facilities.request() as request1:
      yield request1
      yield self.env.process(self.facilities.operation(self))
      self.dataset.patients_operated += 1
      self.facilities.operation_facilities.release(request1)

    rec_q = len(self.facilities.recovery_facilities.queue)
    rec_c = self.facilities.recovery_facilities.count
    self.recovery_queue_length = rec_q
    with self.facilities.recovery_facilities.request(priority=self.treatment_priority) as request2:
      yield request2
      yield self.env.process(self.facilities.recovery(self))
      self.dataset.patients_treated += 1
      self.dataset.add_patient(self)
      self.facilities.recovery_facilities.release(request2)

  def __str__(self):
    return f"Patient no. {self.patients_num}"

  def get_data(self):
    arrival_queue_time = round(self.preparation_start_time - self.arrival_time, 2)
    preparation_time = round(self.preparation_end_time - self.preparation_start_time, 2)
    operation_queue_time = round(self.operation_start_time - self.preparation_end_time, 2)
    operation_time = round(self.operation_end_time - self.operation_start_time, 2)
    recovery_time = round(self.recovery_end_time - self.recovery_start_time, 2)
    total_time = round(self.recovery_end_time - self.arrival_time, 2)
    complication_occurred = float(self.complication_occurred)
    preparation_queue_length = self.preparation_queue_length
    operation_queue_length = self.operation_queue_length
    recovery_queue_length = self.recovery_queue_length
    data = np.array([arrival_queue_time, preparation_time, operation_queue_time, operation_time, recovery_time, total_time, complication_occurred, preparation_queue_length, operation_queue_length, recovery_queue_length])
    return data

  def get_data_headers():
    return ["arrival_queue_time", "preparation_time", "operation_queue_time", "operation_time", "recovery_time", "total_time", "complication_occurred", "preparation_queue_length", "operation_queue_length", "recovery_queue_length", "recovery_full_time", "recovery_queue_length"]


class Dataset:
  """Stores and analyzes the data of all patients that arrive at the hospital. When patient has been treated
  and they are ready to leave, their information is saved into the dataset."""
  _patients = []
  logger = None
  patients_arrived = 0
  patients_prepared = 0
  patients_operated = 0
  patients_treated = 0
  patient_complications = 0
  operation_block_time = 0
  operation_time = 0
  recovery_full_time = 0
  recovery_time = 0

  def __init__(self, verbose=VERBOSE):
    self.logger = Logger(verbose=verbose)
    self.patients_arrived = 0
    self.patients_treated = 0
    self.patients_prepared = 0
    self.patients_operated = 0
    self.patient_complications = 0
    self.operation_block_time = 0
    self.operation_time = 0
    self.recovery_full_time = 0
    self.recovery_time = 0
    self._patients = []

  def add_patient(self, patient):
    self._patients.append(patient)

  def get_dataset(self):
    temp_list = []
    for i in self._patients:
      temp_list.append(list(i.get_data()))

    dataset = np.array(temp_list)
    return dataset

  def analyze(self):
    dataset = self.get_dataset()
    avg_arrival_queue_time = round(np.average(dataset[:,0]),2)
    operation_utilization_time = round(np.sum(dataset[:,3]),2)
    operation_utilization_percentage = round((operation_utilization_time / SIM_TIME)*100, 2)
    average_preparation_queue_length = round(sum(dataset[:,7])/self.patients_arrived, 2)
    average_operation_queue_length = round(sum(dataset[:,8])/self.patients_prepared, 2)
    average_recovery_queue_length = round(sum(dataset[:,9])/self.patients_operated, 2)

    operation_blocked_ratio = round(self.operation_block_time / (self.operation_time + self.operation_block_time), 2)
    recovery_full_ratio = round(self.recovery_full_time/(self.recovery_full_time + self.recovery_time),2)

    self.logger.log(f"Operation was blocked for {operation_blocked_ratio*100}% of the time", 4)
    self.logger.log(f"Average preparation queue length: {average_preparation_queue_length}", 4)
    self.logger.log(f"Complications during operations: {self.patient_complications}", 4)
    self.logger.log(f"Recovery was full for {recovery_full_ratio}% of the time", 4)
    return operation_blocked_ratio, average_operation_queue_length, self.patient_complications, recovery_full_ratio
  
  def __str__(self):
    return f"{self.get_dataset()}"


def hospital(env, dataset, facilities):
  """The main simulation cycle."""

  while True:
    interarrival = random.expovariate(1/PATIENT_INTERARRIVAL_TIME)
    yield env.timeout(interarrival)
    patient = Patient(env, facilities=facilities, dataset=dataset, verbose=VERBOSE)
    env.process(patient.arrival())


def confidence_interval(m, s, t_crit, len_x):
  return (round(m-s*t_crit/np.sqrt(len_x),2), round(m+s*t_crit/np.sqrt(len_x),2))


def main():

  logger = Logger(verbose=VERBOSE)
  logger.log("====================================================================", 1)
  setups = [
    (3, 4, True),
    (3, 5, True),
    (4, 5, True),
    (3, 4, False),
    (3, 5, False),
    (4, 5, False)
  ]


  for (p, r, c) in setups:
    logger.log(f"Simulation with (p={p}, r={r}), random seed = {RANDOM_SEED}, include twist: {c}", 1)
    data = []
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    for i in range(20):
      env = simpy.Environment()
      dataset = Dataset(verbose=VERBOSE)
      facilities = FacilitiesManager(env,verbose=VERBOSE, p=p, r=r, include_twist=c)
      logger.log( f"Simulation no. {i}", 4)
      env.process(hospital(env, dataset=dataset, facilities=facilities))
      env.run(until=SIM_TIME)
      logger.log(f"Patients arrived: {dataset.patients_arrived}", 4)
      logger.log(f"Patients treated: {dataset.patients_treated}", 4)
      op_blocked_ratio, avg_prep_q_len, n_complications, r_full_ratio = dataset.analyze()
      data.append([op_blocked_ratio, avg_prep_q_len, n_complications, r_full_ratio])
    data_array = np.array(data)

    confidence = 0.95
    dof = len(data_array[:,0]) - 1
    len_data = len(data_array[:,0])
    t_crit = np.abs(t.ppf((1-confidence)/2,dof))

    # blocked ratio 
    b_mean = data_array[:,0].mean()
    b_std = data_array[:,0].std()
    b_interval = confidence_interval(m=b_mean, s=b_std, t_crit=t_crit, len_x=len_data)

    # preparation queue length
    p_mean = data_array[:,1].mean()
    p_std = data_array[:,1].std()
    p_interval = confidence_interval(m=p_mean, s=p_std, t_crit=t_crit, len_x=len_data)

    # complications
    c_mean = data_array[:,2].mean()
    c_std = data_array[:,2].std()
    c_interval = confidence_interval(m=c_mean, s=c_std, t_crit=t_crit, len_x=len_data)

    # recovery full ratio
    r_mean = data_array[:,3].mean()
    r_std = data_array[:,3].std()
    r_interval = confidence_interval(m=r_mean, s=r_std, t_crit=t_crit, len_x=len_data)

    logger.log(f"Operation blocked ratio mean: {b_mean:.2f}, 95% confidence interval: {b_interval}", 1)
    logger.log(f"Preparation q length mean: {p_mean:.2f}, 95% confidence interval: {p_interval}", 1)
    logger.log(f"Complications mean: {c_mean:.2f}, 95% confidence interval: {c_interval}", 1)
    logger.log(f"Recovery full ratio mean: {r_mean:.2f}, 95% confidence interval: {r_interval}", 1)
    logger.log("====================================================================", 1)

if __name__ == "__main__":
    main()

