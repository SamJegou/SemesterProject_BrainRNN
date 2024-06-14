# SemesterProject_BrainRNN

This project aims at building a connectome constrained neural network (BrainNN) to train a virtual agent to reproduce a walking behavior.

## How to run the code
### Creation of a graph from the connectome
The BrainNN is initialized from an adjacency matrix of a graph. To create a graph from the connectome, you can run either
- `connectome_by_neurons.py`: it creates a graph from the Standard Leg Premotor Circuit scheme in [Cheong et. al](https://www.biorxiv.org/content/10.1101/2023.06.07.543976v1)
- `connectome_by_target.py`: it will select all the neurons having the target selected in the file ('LegNpT1_L' currently).

The graph created are saved in `data/adj_matrices.json` file. If you have a specific graph, you can directly save it in this file.

### Settings before training the agent
Before training the agents, some settings must be specified:
- in the `settings.py` file, precise which graph you want to use with the `'model'` variable
- if you want to run the BrainNN on its own (not in the RL pipeline), also precise the number of inputs and outputs as well as the batch size.

### Training the agent
The training of the agent is performde by `RL_bipedal_hand.py`. The main parameters are at the begining of the file:
- `TRAIN` is set to True for training, False to load a model and run the enviornment once on it
- `MAX_STEPS`: maximal number of steps in each run
- `N_EPISODE`: number of episodes dunring the training
- `w_I, w_G, w_p, w_v` are the imitation and goal reward respective weights, and the pose and velocity reward respective weights.
The other less important parameters are commented to describe their effects.

You can either run the file directly or parse arguments in a command line. A documentation is written to describe the different arguments, and an example is present in the `exp_wI09_wG01_in02_out02_std01_gamma095_act.run` file.

### Analysing the results
At the end of the training, the actions, rewards, losses, states of the agent during the training are saved in the `save` folder.
They can be analysed by all the `...plots.py` files:
- `reference_plots.py`: plots the reference and trained states, and the reference and trained actions during the last steps of the last epoch (last "non-fallen" epoch)
- `phase_plots.py`: plots some phase graphs (e.g hip angle 1 VS hip angle 2, knee angle 2 VS knee angular velocity 2, etc.)
- `training_plots.py`: plots the rewards, policy-network and value network losses and the angles evolution during the last epoch.

For each of these files, the suffix of the filename must be provided (e.g `_wI09_wG01_in02_out02_std01_gamma095_act` in the example above).
