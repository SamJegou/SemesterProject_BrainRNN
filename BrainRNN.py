import numpy as np

import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from settings import *

def exp_std(w, max_w=np.max(adj_mat), std_lim = 3, a=2):
    return std_lim*np.exp((w/max_w-1)*a)

class BrainRNN(nn.Module):
    def __init__(self, input_size, output_size, adj_mat, layers, activation=torch.sigmoid, batch_size=8, weights_from_connectome='uniform', additive=True, std_fct=exp_std, n_input_nodes=0.2):
        '''
        Arguments
        ---
        input_size: int
            number of inputs to the network (nb of sensory feedback)
        output_size: int
            number of outputs of the nuetwork (nb of actions)
        adj_mat: array of shape (n,n)
            adjacency matrix of the graph, with n neurons/nodes
        layers: list
            index of the nodes contained in each layer of the network
        activation: func
            activation function between all but the output layer
        batch_size: int
            batch size for computations, depreciated
        weigths_from_connectome: str or False
            if False, default weight initialization of torch.nn.Linear
            if str, see `init_connectome_weights` method
        std_fct: func
            function for `init_connectome_weights` method
        n_input_nodes: int, float or None
            if int, number of nodes receiving the input (assigned at random)
            if float, percentage of nodes receiving the input (assigned at random)
            if None, only the first layer receives the input
        '''

        super(BrainRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.adj_mat = adj_mat
        self.n_neurons = len(adj_mat)
        self.layers = layers
        self.activation = activation
        self.batch_size = batch_size
        self.reset_hidden_states()

        self.n_input_nodes=n_input_nodes

        # Create the input layer
        if n_input_nodes is None:
            self.n_input_nodes = None
            self.input_layer = nn.Linear(input_size, len(self.layers[0]))
        else:
            # input layer
            if n_input_nodes<=1:
                self.n_input_nodes=min(int(n_input_nodes*len(adj_mat)), len(adj_mat))
                self.input_layer = nn.Linear(input_size, self.n_input_nodes)
            else:
                self.n_input_nodes=min(n_input_nodes, len(adj_mat))
                self.input_layer = nn.Linear(input_size, self.n_input_nodes)
            self.input_idx = np.random.choice(len(adj_mat), size=self.n_input_nodes, replace=False)
            # create masks for forward use
            self.masks_input = [] # masks[i] = coordinates of input going to layer i
            self.masks_add_vec =  [] # masks[i] = coordinates of add_vec receiving the input
            self.receives_input = [] # wether layer i receives input or not
            for i, layer in enumerate(self.layers):
                mask = []
                for j, node in enumerate(layer): # idx in input for layer i
                    k = np.where(self.input_idx == node)[0]
                    if k.size > 0:
                        mask.append(k)
                if len(mask) > 0:
                    mask = np.hstack(mask)
                self.masks_input.append(mask)

                mask = []
                for j,idx in enumerate(self.input_idx): # coordinates in add_vec receiving input
                    k = np.where(layer==idx)[0]
                    if k.size > 0:
                        mask.append(k)
                if len(mask) > 0:
                    mask = np.hstack(mask)
                    self.receives_input.append(True)
                else:
                    self.receives_input.append(False)
                self.masks_add_vec.append(mask)

        # Create forward hidden layers
        self.hidden_layers = nn.ModuleList([])
        for i in range(len(self.layers)-1): # hidden_layers[i] = layer i to i+1
            new_layer = nn.Linear(len(self.layers[i]),len(self.layers[i+1]))
            mask = self.adj_mat[np.ix_(self.layers[i],self.layers[i+1])] # connections from layer i to layer i+1
            prune.custom_from_mask(new_layer, name='weight', mask=torch.tensor(mask.T)) # delete fictive connections
            # prune.remove(new_layer, 'weight') # makes pruning permanent but removes mask -> trainable weights
            self.hidden_layers.append(new_layer)

        # Create the backward weights
        self.recurrent_layers = nn.ModuleList([]) # recurrent_layers[i](hidden_states) = layer j>i to i
        for i in range(len(self.layers)-1): # last layer should not have incomming reccurent connections
            # Find the backward connections to layer i           
            new_layer = nn.Linear(self.n_neurons, len(self.layers[i]), bias=False) # no bias for backward connection
            mask = self.adj_mat[:,self.layers[i]] # all layers to i
            mask[np.concatenate(layers[:i+1])] = 0 # remove j<=i to i -> keep only j>i to i
            prune.custom_from_mask(new_layer, name='weight', mask=torch.tensor(mask.T)) # delete fictive connections
            # prune.remove(new_layer, 'weight') # makes pruning permanent but removes mask -> trainable weights
            self.recurrent_layers.append(new_layer)
        
        # create skip connections
        nb_nodes = len(self.layers[0]) # nb nodes in 2 first layers
        self.skip_layers = nn.ModuleList([]) # skip_layers[i](layers<i-1) = layer j<=i to i+2
        for i in range(2,len(self.layers)): # first 2 layers should not have skip connections
            # Find the skip connections to layer i
            new_layer = nn.Linear(nb_nodes, len(self.layers[i]), bias=False) # no bias for skip connection
            mask = self.adj_mat[np.concatenate(layers[:i-1])][:,self.layers[i]] # layers j<i-1 to i
            prune.custom_from_mask(new_layer, name='weight', mask=torch.tensor(mask.T)) # delete fictive connections
            # prune.remove(new_layer, 'weight') # makes pruning permanent but removes mask -> trainable weights
            self.skip_layers.append(new_layer)
            nb_nodes += len(self.layers[i-1])

        # Create the output layer
        self.output_layer = nn.Linear(len(self.layers[-1]), output_size)

        if weights_from_connectome:
            self.init_connectome_weights(std_fct=std_fct, law=weights_from_connectome)
    
    def init_connectome_weights(self, std_fct=exp_std, law='uniform', additive=True):
        '''
        Random weight to follow connectome measurments

        std_fct: func
            function of the nb of synapses, that outputs ~std of weight
        law: str
            uniform (-1,1) or normal(0,1), for the law to initialize weights
        additive: bool
            True -> law centered on std_fct(nb_synapses), default std
            False -> law centered on 0 with std = std_fct(nb_synapses)
        '''
        if law == 'uniform':
            rand = (np.random.rand(*self.adj_mat.shape)-1/2)*2
        elif law == 'normal':
            rand = np.random.normal(0,1,self.adj_mat.shape)

        if additive:
            signs = np.random.binomial(n=1, p=0.5, size=self.adj_mat.shape)
            all_weights = signs*exp_std(self.adj_mat)+rand
        else:
            all_weights = exp_std(self.adj_mat)*rand
        #forward layers
        for i in range(len(self.layers)-1):
            layer = self.recurrent_layers[i]
            layer.weight = torch.tensor(all_weights[self.layers[i]][:,self.layers[i]], requires_grad=True)
        # recurrent layers
        for i in range(len(self.layers)-1):
            layer = self.hidden_layers[i]
            weights = all_weights[:,self.layers[i+1]]
            weights[np.concatenate(layers[:i+1])] = 0
            layer.weight = torch.tensor(weights, requires_grad=True)
        # skip layers
        for i in range(2,len(self.layers)):
            layer = self.skip_layers[i-2]
            layer.weight = torch.tensor(all_weights[np.concatenate(layers[:i-1])][:,self.layers[i]], requires_grad=True)

    def forward(self, x):
        next_hidden_states = torch.empty(x.shape[0], self.n_neurons) if x.dim() > 1 else torch.empty(self.n_neurons)
        skips = [] # list of current states for skip connections

        # Input layer
        if self.n_input_nodes is None:
            x = self.activation(self.input_layer(x) + self.recurrent_layers[0](self.hidden_states))
            next_hidden_states[...,self.layers[0]] = x
        else:
            inputs = self.input_layer(x).squeeze()
            add_vec = torch.zeros((x.shape[0],len(self.layers[0]))) if x.dim() > 1 else torch.zeros(len(self.layers[0]))
            add_vec[...,self.masks_add_vec[0]] = inputs[...,self.masks_input[0]]
            x = self.recurrent_layers[0](self.hidden_states) + add_vec
            next_hidden_states[...,self.layers[0]] = x
        

        skips.append(x)

        # "Middle" layers
        for i in range(len(self.layers)-1):
            if self.n_input_nodes is None or not self.receives_input[i]:
                if i == 0: # no skip connection from layer 0 to 1
                    x = self.hidden_layers[i](x) + self.recurrent_layers[i+1](self.hidden_states)
                elif i == len(self.layers)-2: # no recurrent connection for last hidden layer
                    x = self.hidden_layers[i](x) + self.skip_layers[i-1](torch.concat(skips[:-1], dim=-1)) 
                else:
                    x = (self.hidden_layers[i](x) 
                         + self.recurrent_layers[i+1](self.hidden_states)
                         + self.skip_layers[i-1](torch.concat(skips[:-1], dim=-1)) ## skips a pas la bonne dimension
                    )
            else:
                add_vec = torch.zeros((x.shape[0],len(self.layers[i+1]))) if x.dim() > 1 else torch.zeros(len(self.layers[i+1]))
                add_vec[...,self.masks_add_vec[i+1]] = inputs[...,self.masks_input[i+1]]
                if i == 0: # no skip connection from layer 0 to 1
                    x = self.hidden_layers[i](x) + self.recurrent_layers[i+1](self.hidden_states) + add_vec
                elif i == len(self.layers)-2: # no recurrent connection for last hidden layer
                    x = self.hidden_layers[i](x) + self.skip_layers[i-1](torch.concat(skips[:-1], dim=-1)) + add_vec
                else:
                    x = (self.hidden_layers[i](x) 
                         + self.recurrent_layers[i+1](self.hidden_states)
                         + self.skip_layers[i-1](torch.concat(skips[:-1], dim=-1)) ## skips a pas la bonne dimension
                         + add_vec)
            x = self.activation(x)
            next_hidden_states[...,self.layers[i+1]] = x.detach()
            skips.append(x)

        # Output layer
        x = self.output_layer(x) # no activation nor recurrent/skip connection for the last one

        self.hidden_states = next_hidden_states

        return x

    def reset_hidden_states(self, hidden_states=None, hidden_size=None, method='random'):
        if hidden_size is None:
            hidden_size = self.batch_size
        if hidden_states is None:
            if method=='random':
                if hidden_size > 0:
                    self.hidden_states = nn.init.normal_(torch.empty(self.n_neurons), std=1).repeat(hidden_size,1) # same hidden states for all batches
                else:
                    self.hidden_states = nn.init.normal_(torch.empty(self.n_neurons), std=1)
            elif method=='zero':
                if hidden_size > 0:
                    self.hidden_states = nn.init.zeros_(torch.empty(self.n_neurons)).repeat(hidden_size,1) # same hidden states for all batches
                else:
                    self.hidden_states = nn.init.zeros_(torch.empty(self.n_neurons))
            else:
                raise NotImplementedError('Only method "zero" and "random" are implemented.')
        else:
            if self.batch_size > 0:
                assert hidden_states.shape == (hidden_size, self.n_neurons), \
                    f'hidden_states should have shape {(hidden_size, self.n_neurons)}; received {hidden_states.shape}'
            else:
                assert hidden_states.shape == (self.n_neurons,), \
                    f'hidden_states should have shape {(self.n_neurons,)}; received {hidden_states.shape}'
            self.hidden_states = torch.tensor(hidden_states)

def generate_dataset(omega_range, t, n_samples, input_size=1, train_percent=0.7):
    # generation of data
    omegas = np.random.uniform(min(omega_range),max(omega_range),size=n_samples)
    omega_t = np.einsum('i,j->ij',omegas,t) # (n_samples, len(t))
    if input_size == 1:
        x = np.sin(omega_t)
        #y = np.sin(omega_t).reshape((batch_size,n_batch,-1)) # learn identity function
        y = np.mod(omega_t,2*np.pi)
    elif input_size == 2:
        derivative = np.einsum('i,ik->ik', omegas, np.cos(omega_t))
        x = np.dstack((np.sin(omega_t), derivative)) # shape (n_samples, len(t), 2)
        #y = x[::] # identity function
        y = np.dstack((np.mod(omega_t,2*np.pi), derivative))

    # split train/test
    split_idx = int(n_samples*train_percent)
    x_train, x_test = torch.tensor(x[:split_idx,:], dtype=torch.float32), torch.tensor(x[split_idx,::], dtype=torch.float32)
    y_train, y_test = torch.tensor(y[:split_idx,:], dtype=torch.float32), torch.tensor(y[split_idx:,:], dtype=torch.float32)
    
    return (x_train, y_train),(x_test,y_test)

if __name__=='__main__':

    t = np.arange(0,sequence_length*time_step, time_step)
    (x_train, y_train),(x_test,y_test) = generate_dataset(omega_range, t, n_samples, input_size=input_size)
    if normalize:
        x_mean, x_std = np.mean(x_train.numpy(), axis=(0,1)), np.std(x_train.numpy(), axis=(0,1))
        y_mean, y_std = np.mean(y_train.numpy(), axis=(0,1)), np.std(y_train.numpy(), axis=(0,1))
        x_train = torch.tensor((x_train.numpy()-x_mean)/x_std, dtype=torch.float32)
        y_train = torch.tensor((y_train.numpy()-y_mean)/y_std, dtype=torch.float32)
    loader = DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size)

    model = BrainRNN(input_size, output_size, adj_mat, layers, batch_size=batch_size)
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        for i, (x_batch, y_batch) in enumerate(loader):
            model.reset_hidden_states()
            movement = torch.empty(batch_size, sequence_length, output_size)
            for j in range(sequence_length):
                output = model(x_batch[:,j].reshape(-1,input_size))
                #movement[:,j] = output[:,0]
                if output_size == 1:
                    movement[:,j] = torch.remainder(output, 2*np.pi)
                elif output_size == 2:
                    movement[:,j,0] = torch.remainder(output[:,0], 2*np.pi)
                    movement[:,j,1] = output[:,1]
            if shift: # compare out_t and in_(t+1)
                loss = criterion(movement[:,:-1], y_batch.reshape(batch_size, sequence_length, output_size)[:,1:])
            else:
                loss = criterion(movement, y_batch.reshape(batch_size, sequence_length, output_size))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Loss: {loss.item():.4f}"
            )

    torch.save(model.state_dict(), MODEL_PATH)