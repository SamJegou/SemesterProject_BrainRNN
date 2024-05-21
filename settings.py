import json
import numpy as np

DATA_FOLDER = 'data/'
with open(DATA_FOLDER+'adj_matrices.json', 'r') as file:
    dico = json.load(file)


# Model
model = "LegNpT1_L" # "simple", "medium", "standard_leg" (1624 connections) "LegNpT1_L" (37026 connections)

nb_joints = 1
input_params = 24 # observations
output_params = 4 # actions
input_size = input_params*nb_joints
output_size = output_params*nb_joints
pred = 'theta' #'sin'
normalize=False
batch_size = 32
sequence_length = 20
n_samples = 400
shift = True

# Training
num_epochs = 8
learning_rate = 0.05

# Physics & Data
omega_range = [3,7]
t_range = [0,2]
time_step = 0.1
Kp=1
Kd=1


# Derivated variables
MODEL_PATH = f'models/BrainRNN_{model:s}_{input_size:d}_{output_size:d}_sin_{pred:s}.pt'
adj_mat = np.array(dico[model]["matrix"])
layers = dico[model]["layers"]