import numpy as np # version 1.23.3
import simpy # version 4.0.1
# Python version 3.10.6
import argparse

env = simpy.Environment()

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

# Constants of the simulation. Assuming that time is in minutes
NUM_PREPARATION_FACILITIES = 3
NUM_OPERATION_FACILITIES = 1
NUM_RECOVERY_FACILITIES = 3
AVG_PREPARATION_TIME = 40
AVG_OPERATION_TIME = 20
AVG_RECOVERY_TIME = 40
PATIENT_INTERARRIVAL_TIME = 24
SIM_TIME = 60*10
COMPLICATION_RATIO = 0.01 # If complications happen during the operation,
# patient will skip recovery and go to ER instead
DEBUG = True # If true, print logs during the simulation
RANDOM_SEED = None # Set to none if don't want to use seed

np.random.seed(RANDOM_SEED)

# Class for managing empty and reserved facilities
class FacilitiesManager:
  """Managing resources of failicities processing their services"""
  preparation_facilities = None
  operation_facilities = None
  recovery_facilities = None
  env = None
  debug = False

  def __init__(self, env, debug):
    self.env = env
    self.debug = debug
    self.preparation_facilities = simpy.Resource(env, NUM_PREPARATION_FACILITIES)
    self.operation_facilities = simpy.Resource(env, NUM_OPERATION_FACILITIES)
    self.recovery_facilities = simpy.Resource(env, NUM_RECOVERY_FACILITIES)

  def preparation(self, patient):
    if self.debug: print(f"{patient} enters preparation facility at {self.env.now:.2f}")
    patient.preparation_start_time = float(self.env.now)
    random_time = clamp(np.random.normal(AVG_PREPARATION_TIME, 4), AVG_PREPARATION_TIME - 5, AVG_PREPARATION_TIME + 5)
    yield self.env.timeout(random_time)
    if self.debug: print(f"{patient} leaves preparation facility at {self.env.now:.2f}")
    patient.preparation_end_time = float(self.env.now)

  def operation(self, patient):
    if self.debug: print(f"{patient} enters operation facility at {self.env.now:.2f}")
    patient.operation_start_time = float(self.env.now)
    random_time = clamp(np.random.normal(AVG_OPERATION_TIME, 4), AVG_OPERATION_TIME - 5, AVG_OPERATION_TIME + 5)
    yield self.env.timeout(random_time)
    if self.debug: print(f"{patient} leaves operation facility at {self.env.now:.2f}")
    patient.operation_end_time = float(self.env.now)

  def recovery(self, patient):
    if (np.random.uniform(0, 1) <= COMPLICATION_RATIO):
      if self.debug: print(f"complication occurred for {patient}, patient skips recovery")
      patient.complication_occurred = True
      patient.recovery_start_time = float(self.env.now)
      patient.recovery_end_time = float(self.env.now)
      Patient.complications += 1
    else:
      if self.debug: print(f"{patient} enters recovery facility at {self.env.now:.2f}")
      patient.recovery_start_time = float(self.env.now)
      random_time = clamp(np.random.normal(AVG_RECOVERY_TIME, 4), AVG_RECOVERY_TIME - 5, AVG_RECOVERY_TIME + 5)
      yield self.env.timeout(random_time)
      if self.debug: print(f"{patient} has left recovery facility at {self.env.now:.2f}")
      patient.recovery_end_time = float(self.env.now)
      patient.complication_occurred = False

# Patient class to store individual patient's information
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
  debug = False

  def __init__(self, env, debug):
    self.env = env
    self.debug = debug
    Patient.patients_num += 1
    self.patients_num = Patient.patients_num
    if self.debug: print(f"{self} arrives at {env.now:.2f}")
    self.arrival_time = env.now

  def arrival(self, facilities, dataset):
    with facilities.preparation_facilities.request() as request:
      yield request
      yield self.env.process(facilities.preparation(self))

    with facilities.operation_facilities.request() as request1, facilities.recovery_facilities.request() as request2:
      yield request1 & request2
      yield self.env.process(facilities.operation(self))
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
    print("Average arrival queue time:", avg_arrival_queue_time)
    print(f"Operation room utilization time: {operation_utilization_time} ({operation_utilization_percentage} %)")
    print(f"Complications during operations: {Patient.complications}")
  def __str__(self):
    return f"{self.get_dataset()}"

dataset = Dataset()
def hospital(env, debug):
  """The main process of the simulation. Starts the simulation."""
  facilities = FacilitiesManager(env, debug)

  while True:
    yield env.timeout(np.random.randint(PATIENT_INTERARRIVAL_TIME - 1, PATIENT_INTERARRIVAL_TIME + 1))
    patient = Patient(env, debug)
    env.process(patient.arrival(facilities, dataset))

parser = argparse.ArgumentParser()
parser.add_argument('--preparation', metavar='p', required=False, dest='preparation', type=int, help='Number of preparation facilities')
parser.add_argument('--recovery', metavar='r', required=False, dest='recovery', type=int, help="Number of recovery facilities")
args = parser.parse_args()

if args.recovery != None:
  NUM_RECOVERY_FACILITIES = args.recovery
if args.preparation != None: 
  NUM_PREPARATION_FACILITIES = args.preparation

print("Number of recovery facilities:", NUM_RECOVERY_FACILITIES)
print("Number of preparation facilities:", NUM_PREPARATION_FACILITIES)

print("Starting simulation!")
env.process(hospital(env, DEBUG))
env.run(until=SIM_TIME)
print("Simulation ended!") 
print("Patients arrived: ", Patient.patients_num)
print("Patients treated: ", Patient.patients_treated)
print(Patient.get_data_headers())
print(dataset)
dataset.analyze()

