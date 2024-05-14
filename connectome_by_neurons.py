import pandas as pd
import numpy as np
import networkx as nx
import json

PRE_PROCESS = True
PROCESS = True
DROP_EDGES = 5 # delete connections with less (strict) than 5 synapses
TARGET = 'LegNpT1_L'

DATA_FOLDER = 'data/'
CONNECTION_FOLDER = 'manc-traced-adjacencies-v1.0/'
CONNECTIONS = DATA_FOLDER + CONNECTION_FOLDER + 'traced-connections.csv'
PROPERTIES = DATA_FOLDER + 'manc-v1.0-neuron-properties.feather'
ROI = DATA_FOLDER+CONNECTION_FOLDER+"traced-connections-per-roi.csv"
MN_GROUPS = DATA_FOLDER+'Serial Leg MN groups.csv'
LEG_GROUPS = DATA_FOLDER+'Standard Leg groups.csv'
STANDARD_LEG_CIRCUIT = DATA_FOLDER+'standard_leg_premotor_circuit.txt'

SAVE_NAME_DF = 'processed_standard_leg_groups.csv'
SAVE_NAME_ADJ_MAT = 'adj_mat_standard_leg.npy'
SAVE_NAME_LAYERS = 'layers_standard_leg.json'

connections = pd.read_csv(CONNECTIONS)
properties = pd.read_feather(PROPERTIES)
df_roi = pd.read_csv(ROI)
leg_groups = pd.read_csv(LEG_GROUPS)
mn_groups = pd.read_csv(MN_GROUPS)

print('Data loaded')
# changes 'LegNp(T2)(L)' to 'LegNpT2_L'
df_roi["roi"] = df_roi["roi"].apply(lambda s: s.replace(')(','_').replace(')','').replace('(',''))


with open(STANDARD_LEG_CIRCUIT, 'r') as file:
    neuron_types = file.readlines()
neuron_types = list(map(lambda s:s.rstrip(), neuron_types)) # removes '\n'

if PRE_PROCESS:
    dico = {}

    for i, name in enumerate(neuron_types):
        if name == '': # motorneurons will follow
            break
        elif '_' in name: # this is a serial set
            group = leg_groups[leg_groups["plot_name"] == name]
            for idx, serie in group.iterrows():
                dico[serie["bodyid"]] = {
                    "type":serie["type"],
                    "group_name":name,
                    "target":[serie["target"]]
                }
        else: # just a type of neuron
            neuron_type = name.split('(')[0]
            ids = list(properties[properties["type"] == neuron_type]["bodyId"])
            for id in ids:
                target = properties[properties["bodyId"] == id]["outputRois"]
                target = list(map(lambda s: s.replace(')(','_').replace(')','').replace('(',''), list(target)[0]))
                dico[id] = {
                    "type": neuron_type,
                    "group_name":None,
                    "target":target
                }
    for name in neuron_types[i+1:]: # motorneurons
        neuron_type = name.split('(')[0]
        ids = list(properties[properties["type"] == neuron_type]["bodyId"])
        for id in ids:
            target = properties[properties["bodyId"] == id]["outputRois"]
            target = list(map(lambda s: s.replace(')(','_').replace(')','').replace('(',''), list(target)[0]))
            dico[id] = {
                "type": neuron_type,
                "group_name":None,
                "target":target
            }

    output = pd.DataFrame.from_dict(
        dico,
        orient='index',
        columns=['type', 'group_name', "target"]
    )
    output.reset_index(inplace=True) # set ids as a columns
    output.rename(columns={'index':'id'}, inplace=True) # rename 'index' into 'id'
    output.sort_values(by='id', inplace=True) # sort by increasing id
    
    output.to_csv(DATA_FOLDER+SAVE_NAME_DF, index=False)
    print('Relevant neurons retrieved...')
else:
    output = pd.read_csv(DATA_FOLDER+SAVE_NAME_DF)

mask = output.target.apply(lambda x: TARGET in x)
output_target = output[mask] # neurons with appropriate neuron types & target
output_target.reset_index(drop=True, inplace=True) # re-index rows
MN_idx = list(output_target[output_target["type"].str.contains('MN')].index)
all_ids = list(output_target.id)

if PRE_PROCESS:
    short_connections = connections[
        (connections.bodyId_pre.isin(all_ids)) 
        & (connections.bodyId_post.isin(all_ids))] # select connections between neurons of interest
    adj_mat = np.zeros((len(output_target), len(output_target)), dtype=np.uint16)
    for i, serie in output_target.iterrows():
        id = serie["id"]
        # Retrieve connected neurons' indexes in adjacency matrix
        out_connections = short_connections[short_connections["bodyId_pre"] == id]
        if out_connections.empty:
            print(f'Id={id:d} not found in internal connections')
            print(f'Neuron type: {serie["type"]:s}')
        out_ids = out_connections.bodyId_post.values
        out_weights = np.array(out_connections.weight)
        out_idxs = list(output_target[output_target.id.isin(out_ids)].index)

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

    # find/define first layer (receiving input)
    first_neuron = neuron_types[0].split('(')[0]
    idx_first_layer = output_target[output_target["type"] == first_neuron].index[0]
    first_color = d[idx_first_layer]

    # group same color nodes by layers
    colors = ([first_color]+
              [i for i in range(nb_colors) if i!= first_color]
    )
    layers = [[] for _ in range(nb_colors)]
    for idx, color in d.items():
        layers[color].append(idx)
    
    # last layer of motorneurons
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
dico["standard_leg"] = {
    "layers": layers,
    "matrix":adj_mat.tolist()
}
with open(DATA_FOLDER+'adj_matrices.json', 'w') as file:
    json.dump(dico, file)