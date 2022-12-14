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

  prep_idle_start_time = None
  prep_idle_end_time = None
  op_idle_start_time = None
  op_idle_end_time = None
  rec_idle_start_time = None
  rec_idle_end_time = None
  
  prep_idle_time = None
  op_idle_time = None
  rec_idle_time = None

  def stop_prep_idle(self):
    if self.prep_idle_start_time != None:
      self.prep_idle_end_time = self.env.now
      idle_time = round(self.prep_idle_end_time - self.prep_idle_start_time, 2)
      self.prep_idle_time += idle_time
      self.prep_idle_start_time = None
    else:
      raise Exception("Preparation facilities were not idling")
    
  def __init__(
      self, 
      env, 
      verbose=VERBOSE, 
      p = N_PREPARATION_FACILITIES, 
      o = N_OPERATION_FACILITIES, 
      r = N_RECOVERY_FACILITIES, 
      prep_time = PREPARATION_TIME, 
      op_time = OPERATION_TIME, 
      rec_time = RECOVERY_TIME, 
      compl_chance = COMPLICATION_CHANGE, 
      include_twist = True
    ):
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

    self.prep_idle_start_time = self.env.now
    self.prep_idle_end_time = None
    self.op_idle_start_time = None
    self.op_idle_end_time = None
    self.rec_idle_start_time = None
    self.rec_idle_end_time = None

  def preparation(self, patient):
    self.logger.log(f"{patient} enters preparation facility at {self.env.now:.2f}", 5)    
    random_time = random.expovariate(1/self.prep_time)
    self.logger.log(f"{patient} preparation time is {random_time}", 5)
    yield self.env.timeout(random_time)
    self.logger.log(f"  {patient} leaves preparation facility at {self.env.now:.2f}", 5)

  def operation(self, patient):
    self.logger.log(f"{patient} enters operation facility at {self.env.now:.2f}", 5)
    random_time = random.expovariate(1/self.op_time)
    self.logger.log(f"{patient} operation time is {random_time}")
    yield self.env.timeout(random_time)
    if (self.include_twist and random.uniform(0, 1) <= self.compl_ratio):
      self.logger.log(f"Complication occurred for {patient}, patient is put to high priority recovery queue", 5)
      patient.treatment_priority = 1
      patient.dataset.patient_complications += 1
    self.logger.log(f"  {patient} leaves operation facility at {self.env.now:.2f}", 5)

  def recovery(self, patient):
    self.logger.log(f"{patient} enters recovery facility at {self.env.now:.2f}", 5)
    recovery_time_multiplier = 1
    if patient.treatment_priority == 1:
      recovery_time_multiplier = 2
    random_time = random.expovariate(1/self.rec_time*recovery_time_multiplier)
    self.logger.log(f"{patient} recovery time is {random_time}")
    yield self.env.timeout(random_time)
    self.logger.log(f"  {patient} leaves recovery facility at {self.env.now:.2f}", 5)

class Patient:
  """Manages single patient's life cycle. Stores data of individual patient and keeps track of 
  all arrived patients and treated patients"""
  patients_num = 0
  treatment_priority = 10 # Default priority
  arrival_time = None
  dataset = None
  facilities = None
  env = None
  logger = None

  def __init__(self, env, facilities, dataset, verbose=VERBOSE):
    self.treatment_priority = 10
    self.patients_num = 0
    self.arrival_time = None
    self.dataset = dataset
    self.facilities = facilities
    self.logger = Logger(verbose)
    self.env = env
    self.dataset.patients_arrived += 1
    self.patients_num = self.dataset.patients_arrived
    self.logger.log(f"{self} arrives at {env.now:.2f}", 5)
    self.arrival_time = float(env.now)

  def arrival(self):
    prep_q_len = len(self.facilities.preparation_facilities.queue)
    self.dataset.preparation_queue_lengths.append(prep_q_len)
    with self.facilities.preparation_facilities.request() as in_prep_room:
      yield in_prep_room
      yield self.env.process(self.facilities.preparation(self))
      self.facilities.preparation_facilities.release(in_prep_room)

    op_q = len(self.facilities.operation_facilities.queue)
    self.operation_queue_length = op_q
    with self.facilities.operation_facilities.request() as in_op_room, self.facilities.recovery_facilities.request(priority=self.treatment_priority) as in_rec_room:
      yield in_op_room
      operation_time_start = self.env.now
      yield self.env.process(self.facilities.operation(self))
      block_start_time = self.env.now
      rec_q = len(self.facilities.recovery_facilities.queue)
      self.recovery_queue_length = rec_q
      yield in_rec_room
      block_end_time = self.env.now
      self.facilities.operation_facilities.release(in_op_room)
      block_time = round(block_end_time - block_start_time, 2)
      operation_time = round(block_end_time - operation_time_start, 2)
      self.dataset.operation_blocked_time += block_time
      self.dataset.operation_time += operation_time
      yield self.env.process(self.facilities.recovery(self))
      self.facilities.recovery_facilities.release(in_rec_room)
      self.dataset.patients_treated += 1

  def __str__(self):
    return f"Patient no. {self.patients_num}"

  def get_data(self):
    complication_occurred = float(self.complication_occurred)
    data = np.array([self.arrival_queue_time, self.preparation_time, self.operation_queue_time, self.operation_time, self.recovery_time, self.total_time, complication_occurred, self.preparation_queue_length, self.operation_queue_length, self.recovery_queue_length])
    return data

  def get_data_headers():
    return ["arrival_queue_time", "preparation_time", "operation_queue_time", "operation_time", "recovery_time", "total_time", "complication_occurred", "preparation_queue_length", "operation_queue_length", "recovery_queue_length"]


class Dataset:
  """Stores and analyzes the data of all patients that arrive at the hospital. When patient has been treated
  and they are ready to leave, their information is saved into the dataset."""
  _patients = []
  logger = None

  patients_arrived = 0
  patients_treated = 0

  preparation_queue_lengths = []
  operation_blocked_time = 0
  operation_time = 0

  patient_complications = 0

  def __init__(self, verbose=VERBOSE):
    self.logger = Logger(verbose=verbose)

    self.patients_arrived = 0
    self.patients_treated = 0

    self.preparation_queue_lengths = []
    self.operation_blocked_time = 0
    self.operation_time = 0
    
    self.patient_complications = 0
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
    op_block_ratio = round(self.operation_blocked_time / self.operation_time, 2)
    avg_prep_q_len = sum(self.preparation_queue_lengths)/len(self.preparation_queue_lengths)
    complications = self.patient_complications
    self.logger.log(f"Operation was blocked for {op_block_ratio} of the time", 4)
    self.logger.log(f"Average preparation queue length: {avg_prep_q_len}", 4)
    self.logger.log(f"Complications during operations: {complications}", 4)
    return op_block_ratio, avg_prep_q_len, complications
  
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
      op_blocked_ratio, avg_prep_q_len, n_complications = dataset.analyze()
      data.append([op_blocked_ratio, avg_prep_q_len, n_complications])
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

    logger.log(f"Operation blocked ratio mean: {b_mean:.2f}, 95% confidence interval: {b_interval}", 1)
    logger.log(f"Preparation q length mean: {p_mean:.2f}, 95% confidence interval: {p_interval}", 1)
    logger.log(f"Complications mean: {c_mean:.2f}, 95% confidence interval: {c_interval}", 1)
    logger.log("====================================================================", 1)

if __name__ == "__main__":
    main()

