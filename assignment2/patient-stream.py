import numpy as np # version 1.23.3
import simpy # version 4.0.1
# Python version 3.10.6
import random

# Constants of the simulation. Assuming that time is in minutes
N_PREPARATION_FACILITIES = 3
N_OPERATION_FACILITIES = 1
N_RECOVERY_FACILITIES = 3
PREPARATION_TIME = 40
OPERATION_TIME = 20
RECOVERY_TIME = 40
PATIENT_INTERARRIVAL_TIME = 24
SIM_TIME = 1000
COMPLICATION_CHANGE = 0.01 # If complications happen during the operation, patient will skip recovery and go to ER instead
RANDOM_SEED = None # Set to none if don't want to use seed
VERBOSE = 5

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

logger = Logger(VERBOSE)
env = simpy.Environment()

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

  def __init__(self, env, p = N_PREPARATION_FACILITIES, o = N_OPERATION_FACILITIES, r = N_RECOVERY_FACILITIES, prep_time = PREPARATION_TIME, op_time = OPERATION_TIME, rec_time = RECOVERY_TIME, compl_chance = COMPLICATION_CHANGE):
    self.env = env
    self.preparation_facilities = simpy.Resource(env, p)
    self.operation_facilities = simpy.Resource(env, o)
    self.recovery_facilities = simpy.Resource(env, r)
    self.prep_time = prep_time
    self.op_time = op_time
    self.rec_time = rec_time
    self.compl_ratio = compl_chance

  def preparation(self, patient):
    logger.log(f"{patient} enters preparation facility at {self.env.now:.2f}", 5)
    patient.preparation_start_time = float(self.env.now)
    random_time = random.expovariate(self.prep_time)
    yield self.env.timeout(random_time)
    logger.log(f"{patient} leaves preparation facility at {self.env.now:.2f}", 5)
    patient.preparation_end_time = float(self.env.now)

  def operation(self, patient):
    logger.log(f"{patient} enters operation facility at {self.env.now:.2f}", 5)
    patient.operation_start_time = float(self.env.now)
    random_time = random.expovariate(self.op_time)
    yield self.env.timeout(random_time)
    logger.log(f"{patient} leaves operation facility at {self.env.now:.2f}", 5)
    patient.operation_end_time = float(self.env.now)

  def recovery(self, patient):
    if (np.random.uniform(0, 1) <= self.compl_ratio):
      logger.log(f"complication occurred for {patient}, patient skips recovery", 5)
      patient.complication_occurred = True
      patient.recovery_start_time = float(self.env.now)
      patient.recovery_end_time = float(self.env.now)
      Patient.complications += 1
    else:
      logger.log(f"{patient} enters recovery facility at {self.env.now:.2f}", 5)
      patient.recovery_start_time = float(self.env.now)
      random_time = random.expovariate(self.rec_time)
      yield self.env.timeout(random_time)
      logger.log(f"{patient} has left recovery facility at {self.env.now:.2f}", 5)
      patient.recovery_end_time = float(self.env.now)
      patient.complication_occurred = False


class Patient:
  """Manages single patient's life cycle. Stores data of individual patient and keeps track of 
  all arrived patients and treated patients"""
  patients_num = 0
  patients_treated = 0
  complications = 0
  arrival_time = None
  preparation_start_time = None
  preparation_end_time = None
  operation_start_time = None
  operation_end_time = None
  recovery_start_time = None
  recovery_end_time = None
  complication_occurred = None
  env = None

  def __init__(self, env):
    self.env = env
    Patient.patients_num += 1
    self.patients_num = Patient.patients_num
    logger.log(f"{self} arrives at {env.now:.2f}", 5)
    self.arrival_time = env.now

  def arrival(self, facilities, dataset):
    with facilities.preparation_facilities.request() as request:
      yield request
      yield self.env.process(facilities.preparation(self))

    with facilities.operation_facilities.request() as request1, facilities.recovery_facilities.request() as request2:
      yield request1
      yield self.env.process(facilities.operation(self))

    with facilities.recovery_facilities.request() as request2:
      yield request2
      yield self.env.process(facilities.recovery(self))
      Patient.patients_treated += 1
      dataset.add_patient(self)

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
    data = np.array([arrival_queue_time, preparation_time, operation_queue_time, operation_time, recovery_time, total_time, complication_occurred])
    return data

  def get_data_headers():
    return ["arrival_queue_time", "preparation_time", "operation_queue_time", "operation_time", "recovery_time", "total_time", "complication_occurred"]


class Dataset:
  """Stores and analyzes the data of all patients that arrive at the hospital. When patient has been treated
  and they are ready to leave, their information is saved into the dataset."""
  _patients = []

  def add_patient(self, patient):
    Dataset._patients.append(patient)

  def get_dataset(self):
    temp_list = []
    for i in Dataset._patients:
      temp_list.append(list(i.get_data()))

    dataset = np.array(temp_list)
    return dataset

  def analyze(self):
    dataset = self.get_dataset()
    avg_arrival_queue_time = round(np.average(dataset[:,0]),2)
    operation_utilization_time = round(np.sum(dataset[:,3]),2)
    operation_utilization_percentage = round((operation_utilization_time / SIM_TIME)*100, 2)
    logger.log(f"Average arrival queue time: {avg_arrival_queue_time}", 1)
    logger.log(f"Operation room utilization time: {operation_utilization_time} ({operation_utilization_percentage} %)", 1)
    logger.log(f"Complications during operations: {Patient.complications}", 1)
  def __str__(self):
    return f"{self.get_dataset()}"


dataset = Dataset()
def hospital(env):
  """The main process of the simulation. Starts the simulation."""
  facilities = FacilitiesManager(env)

  while True:
    interarrival = random.expovariate(PATIENT_INTERARRIVAL_TIME)
    yield env.timeout(interarrival)
    patient = Patient(env)
    env.process(patient.arrival(facilities, dataset))


def main():
  logger.log("Starting simulation!", 1)
  env.process(hospital(env))
  env.run(until=SIM_TIME)
  logger.log("Simulation ended!", 1) 
  logger.log(f"Patients arrived: {Patient.patients_num}", 1)
  logger.log(f"Patients treated: {Patient.patients_treated}", 1)
  print(Patient.get_data_headers())
  print(dataset)
  dataset.analyze()

if __name__ == "__main__":
    main()

