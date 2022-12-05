import numpy as np # version 1.23.3
import simpy # version 4.0.1
# Python version 3.10.6
import random
from scipy.stats import t
import math

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
      include_twist = True,
      prep_time_unif = None,
      op_time_unif = None,
      rec_time_unif = None,
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

    self.prep_time_unif = prep_time_unif
    self.op_time_unif = op_time_unif
    self.rec_time_unif = rec_time_unif

  def preparation(self, patient):
    self.logger.log(f"{patient} enters preparation facility at {self.env.now:.2f}", 5)    
    random_time = random.expovariate(1/self.prep_time) if self.prep_time_unif == None else random.uniform(self.prep_time_unif[0], self.prep_time_unif[1])
    self.logger.log(f"{patient} preparation time is {random_time}", 5)
    yield self.env.timeout(random_time)
    self.logger.log(f"  {patient} leaves preparation facility at {self.env.now:.2f}", 5)

  def operation(self, patient):
    self.logger.log(f"{patient} enters operation facility at {self.env.now:.2f}", 5)
    random_time = random.expovariate(1/self.op_time) if self.op_time_unif == None else random.uniform(self.op_time_unif[0], self.op_time_unif[1])
    self.logger.log(f"{patient} operation time is {random_time}")
    yield self.env.timeout(random_time)
    if (self.include_twist and np.random.uniform(0, 1) <= self.compl_ratio):
      self.logger.log(f"Complication occurred for {patient}, patient is put to high priority recovery queue", 5)
      patient.treatment_priority = 1
      patient.dataset.patient_complications += 1
      patient.complication_occurred = True
    self.logger.log(f"  {patient} leaves operation facility at {self.env.now:.2f}", 5)

  def recovery(self, patient):
    self.logger.log(f"{patient} enters recovery facility at {self.env.now:.2f}", 5)
    recovery_time_multiplier = 1
    if patient.treatment_priority == 1:
      recovery_time_multiplier = 2
    random_time = random.expovariate(1/self.rec_time*recovery_time_multiplier) if self.rec_time_unif == None else random.uniform(self.rec_time_unif[0], self.rec_time_unif[1])
    self.logger.log(f"{patient} recovery time is {random_time}")
    yield self.env.timeout(random_time)
    self.logger.log(f"  {patient} leaves recovery facility at {self.env.now:.2f}", 5)

class Patient:
  """Manages single patient's life cycle. Stores data of individual patient and keeps track of 
  all arrived patients and treated patients"""
  patients_num = 0
  treatment_priority = 10 # Default priority
  dataset = None
  facilities = None
  env = None
  logger = None

  arrival_time = None
  operation_blocked_time = None
  preparation_queue_length = None
  operation_time = None
  total_time = None
  complication_occurred = False

  def __init__(self, env, facilities, dataset, verbose=VERBOSE):
    self.treatment_priority = 10
    self.patients_num = 0

    self.arrival_time = None
    self.operation_blocked_time = None
    self.preparation_queue_length = None
    self.operation_time = None
    self.total_time = None
    self.complication_occurred = False

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
    self.preparation_queue_length = prep_q_len
    with self.facilities.preparation_facilities.request() as in_prep_room:
      yield in_prep_room
      yield self.env.process(self.facilities.preparation(self))
      self.facilities.preparation_facilities.release(in_prep_room)

    op_q = len(self.facilities.operation_facilities.queue)
    self.operation_queue_length = op_q
    with self.facilities.operation_facilities.request() as in_op_room, self.facilities.recovery_facilities.request(priority=self.treatment_priority) as in_rec_room:
      yield in_op_room
      operation_time_start = float(self.env.now)
      yield self.env.process(self.facilities.operation(self))
      block_start_time = float(self.env.now)
      rec_q = len(self.facilities.recovery_facilities.queue)
      self.recovery_queue_length = rec_q
      yield in_rec_room
      block_end_time = float(self.env.now)
      self.facilities.operation_facilities.release(in_op_room)
      block_time = block_end_time - block_start_time
      operation_time = block_end_time - operation_time_start
      self.operation_blocked_time = block_time
      self.operation_time = operation_time
      self.dataset.operation_blocked_time += block_time
      self.dataset.operation_time += operation_time
      yield self.env.process(self.facilities.recovery(self))
      self.facilities.recovery_facilities.release(in_rec_room)
      self.dataset.patients_treated += 1
      self.total_time = float(float(self.env.now) - self.arrival_time)
      self.dataset.add_patient(self)

  def __str__(self):
    return f"Patient no. {self.patients_num}"

  def get_data(self):
    complication_occurred = float(self.complication_occurred)
    data = np.array([self.arrival_time,  self.preparation_queue_length, self.operation_blocked_time, self.operation_time, self.operation_time, complication_occurred, self.total_time])
    return data

  def get_data_headers():
    return ["arrival_time",  "preparation_queue_length", "operation_blocked_time", "operation_time", "operation_time", "complication_occurred", "total_time"]


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

  # Crude/fast way to split dataset to samples while leaving some unused data in-between the samples
  def split_dataset(self, splits):
    dataset = self.get_dataset()
    chunks = (splits*2)+1
    num_elements = math.floor(len(dataset)/chunks)
    splits = []
    for i in range(0, chunks):
      split = dataset[i*num_elements:min(i*num_elements+num_elements, len(dataset))]
      splits.append(split)

    splitted = splits[1::2]
    return splitted

  def analyze(self):
    op_block_ratio = round(self.operation_blocked_time / self.operation_time, 2)
    avg_prep_q_len = round(sum(self.preparation_queue_lengths)/len(self.preparation_queue_lengths),2)
    complications = self.patient_complications
    self.logger.log(f"Operation was blocked for {op_block_ratio*100}% of the time", 1)
    self.logger.log(f"Average preparation/entrance queue length: {avg_prep_q_len}", 1)
    self.logger.log(f"Complications during operations: {complications}", 1)
    return op_block_ratio, avg_prep_q_len, complications
  
  def __str__(self):
    return f"{self.get_dataset()}"


def hospital(env, dataset, facilities, interarrival_time=PATIENT_INTERARRIVAL_TIME, interarrival_time_unif=None):
  """The main simulation cycle."""

  while True:
    interarrival = random.expovariate(1/interarrival_time) if interarrival_time_unif == None else random.uniform(interarrival_time_unif[0], interarrival_time_unif[1])
    yield env.timeout(interarrival)
    patient = Patient(env, facilities=facilities, dataset=dataset, verbose=VERBOSE)
    env.process(patient.arrival())


def confidence_interval(m, s, t_crit, len_x):
  return (round(m-s*t_crit/np.sqrt(len_x),2), round(m+s*t_crit/np.sqrt(len_x),2))


def main():

  np.set_printoptions(suppress=True)
  logger = Logger(verbose=VERBOSE)
  logger.log("====================================================================", 1)

  configurations = [
    dict(p=2, r=4, prep_time=40, prep_time_unif=None, rec_time=40, rec_time_unif=None, inter_time=25, inter_time_unif=None, compl_chance=0.05),
    dict(p=2, r=4, prep_time=None, prep_time_unif=(30,50), rec_time=None, rec_time_unif=(30,50), inter_time=25, inter_time_unif=None, compl_chance=0.05),
    dict(p=2, r=4, prep_time=40, prep_time_unif=None, rec_time=40, rec_time_unif=None, inter_time=None, inter_time_unif=(20,30), compl_chance=0.05),
    dict(p=2, r=4, prep_time=None, prep_time_unif=(30,50), rec_time=None, rec_time_unif=(30,50), inter_time=None, inter_time_unif=(20,30), compl_chance=0.05),
    dict(p=2, r=4, prep_time=40, prep_time_unif=None, rec_time=40, rec_time_unif=None, inter_time=22.5, inter_time_unif=None, compl_chance=0.05),
    dict(p=2, r=4, prep_time=None, prep_time_unif=(30,50), rec_time=None, rec_time_unif=(30,50), inter_time=22.5, inter_time_unif=None, compl_chance=0.05),
    dict(p=2, r=4, prep_time=40, prep_time_unif=None, rec_time=40, rec_time_unif=None, inter_time=None, inter_time_unif=(20,25), compl_chance=0.05),
    dict(p=2, r=4, prep_time=None, prep_time_unif=(30,50), rec_time=None, rec_time_unif=(30,50), inter_time=None, inter_time_unif=(20,25), compl_chance=0.05),
  ]

  for conf in configurations:
    sim_time = 10000
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    logger.log(f"Simulation with (p={conf['p']}, r={conf['r']}), random seed = {RANDOM_SEED}, include twist: {True}, prep_time: {conf['prep_time'] if conf['prep_time_unif'] == None else conf['prep_time_unif']}, rec_time: {conf['rec_time'] if conf['rec_time_unif'] == None else conf['rec_time_unif']}, inter_time: {conf['inter_time'] if conf['inter_time_unif'] == None else conf['inter_time_unif']}, compl_chance:{conf['compl_chance']}", 1)
    dataset = Dataset(verbose=VERBOSE)
    facilities = FacilitiesManager(
      env,verbose=VERBOSE, 
      p=conf['p'], 
      r=conf['r'], 
      include_twist=True, 
      prep_time=conf['prep_time'], 
      rec_time=conf['rec_time'], 
      prep_time_unif=conf['prep_time_unif'], 
      rec_time_unif=conf['rec_time_unif'],
      compl_chance=conf['compl_chance']
      )
    env.process(hospital(env, 
      dataset=dataset, 
      facilities=facilities, 
      interarrival_time=conf['inter_time'], 
      interarrival_time_unif=conf['inter_time_unif']
    ))
    env.run(until=sim_time)

    samples = dataset.split_dataset(10)
    correlations = []

    for i in range(1, len(samples)):
      correlation = 0
      X = samples[i-1][:,1]
      Y = samples[i][:,1]
      X_std = np.std(X)
      Y_std = np.std(Y)
      if X_std != 0 and Y_std != 0:
        # Get the Pearson correlation coefficient
        correlation = np.corrcoef(X, Y)[1][0]
      correlations.append(round(correlation,4))

    logger.log(f"correlations between adjacent samples:\n{correlations}", 1)
    dataset.analyze()

    logger.log("====================================================================", 1)

if __name__ == "__main__":
    main()

