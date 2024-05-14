import pandas as pd
import numpy as np
import networkx as nx
import json

PRE_PROCESS = True # compute adj_mat
PROCESS = True # build graph and layers
DROP_EDGES = 5 # delete connections with less (strict) than 5 synapses

TARGET = 'LegNpT1_L'
FILTER_IN = True
FIRST_NEURON_TYPE = None # or string

DATA_FOLDER = 'data/'
CONNECTION_FOLDER = 'manc-traced-adjacencies-v1.0/'
CONNECTIONS = DATA_FOLDER + CONNECTION_FOLDER + 'traced-connections.csv'
PROPERTIES = DATA_FOLDER + 'manc-v1.0-neuron-properties.feather'
ROI = DATA_FOLDER+CONNECTION_FOLDER+"traced-connections-per-roi.csv"
MN_GROUPS = DATA_FOLDER+'Serial Leg MN groups.csv'
LEG_GROUPS = DATA_FOLDER+'Standard Leg groups.csv'
STANDARD_LEG_CIRCUIT = DATA_FOLDER+'standard_leg_premotor_circuit.txt'

SAVE_NAME_ADJ_MAT = f'adj_mat_{TARGET:s}.npy'
SAVE_NAME_LAYERS = f'layers_{TARGET:s}.json'




connections = pd.read_csv(CONNECTIONS)
properties = pd.read_feather(PROPERTIES)
df_roi = pd.read_csv(ROI)
leg_groups = pd.read_csv(LEG_GROUPS)
mn_groups = pd.read_csv(MN_GROUPS)

print('Data loaded')
# changes 'LegNp(T2)(L)' to 'LegNpT2_L'
df_roi["roi"] = df_roi["roi"].apply(lambda s: s.replace(')(','_').replace(')','').replace('(',''))

## With ROIs
#properties["outputRois"] = properties["outputRois"].apply(lambda l: [s.replace(')(','_').replace(')','').replace('(','') for s in l])
#mask = properties.outputRois.apply(lambda x: TARGET in x)
#properties_target = properties[mask] # neurons with appropriate neuron types & target

## With target
properties_target = properties[properties.target == TARGET]

if FILTER_IN:
    properties_target = properties_target[(properties_target["type"].str.contains('IN')) | (properties_target["type"].str.contains('MN'))]
properties_target.reset_index(drop=True, inplace=True) # re-index rows
MN_idx = list(properties_target[properties_target["type"].str.contains('MN')].index)
all_ids = list(properties_target.bodyId)

if PRE_PROCESS:
    short_connections = connections[
        (connections.bodyId_pre.isin(all_ids)) 
        & (connections.bodyId_post.isin(all_ids))] # select connections between neurons of interest
    adj_mat = np.zeros((len(properties_target), len(properties_target)), dtype=np.uint16)
    for i, serie in properties_target.iterrows():
        id = serie["bodyId"]
        # Retrieve connected neurons' indexes in adjacency matrix
        out_connections = short_connections[short_connections["bodyId_pre"] == id]
        if out_connections.empty:
            print(f'Id={id:d} not found in internal connections')
            print(f'Neuron type: {serie["type"]:s}')
        out_ids = out_connections.bodyId_post.values
        out_weights = np.array(out_connections.weight)
        out_idxs = list(properties_target[properties_target.bodyId.isin(out_ids)].index)

        adj_mat[i, out_idxs] = out_weights
    adj_mat[adj_mat < DROP_EDGES] = 0

    with open(DATA_FOLDER+SAVE_NAME_ADJ_MAT, 'wb') as file:
        np.save(file, adj_mat)
    print('Adjacency matrix computed.')
else:
    adj_mat = np.load(DATA_FOLDER+SAVE_NAME_ADJ_MAT)

if PROCESS: # organization in layers
    mask = np.delete(np.arange(len(all_ids)), MN_idx) # all non-MN neurons
    masked_adj_mat = adj_mat[mask,:]
    masked_adj_mat = masked_adj_mat[:,mask]
    graph = nx.from_numpy_array(masked_adj_mat) # motorneurons will be placed at the end
    d = nx.coloring.greedy_color(graph, strategy="largest_first")
    nb_colors = np.max(list(d.values())) + 1

    if FIRST_NEURON_TYPE is not None: # find/define first layer (receiving input)
        idx_first_layer = properties_target[properties_target["type"] == FIRST_NEURON_TYPE].index[0]
        first_color = d[idx_first_layer]

        # group same color nodes by layers
        colors = ([first_color]+
                  [i for i in range(nb_colors) if i!= first_color]
        )
    else:
        colors = list(range(nb_colors))

    layers = [[] for _ in range(nb_colors)]
    for idx, color in d.items():
        layers[color].append(idx)
    
    # last layer of motorneurons
    if len(MN_idx) != 0: #pbm of 0 MN
        layers.append(MN_idx)
    
    for layer in layers:
        layer.sort()
    
    with open(DATA_FOLDER+SAVE_NAME_LAYERS, 'w') as file:
        json.dump(layers, file)
    print('Graph organized in layers')
else:
    with open(DATA_FOLDER+SAVE_NAME_LAYERS, 'r') as file:
        layers = json.load(file)

pass
with open(DATA_FOLDER+'adj_matrices.json', 'r') as file:
    dico = json.load(file)
dico[TARGET] = {
    "layers": layers,
    "matrix":adj_mat.tolist()
}
with open(DATA_FOLDER+'adj_matrices.json', 'w') as file:
    json.dump(dico, file)