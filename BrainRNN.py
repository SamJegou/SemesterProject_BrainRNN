import numpy as np

import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from settings import *

class BrainRNN(nn.Module):
    def __init__(self, input_size, output_size, adj_mat, layers, activation=torch.sigmoid, batch_size=8):
        super(BrainRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.adj_mat = adj_mat
        self.n_neurons = len(adj_mat)
        self.layers = layers
        self.activation = activation
        self.batch_size = batch_size
        self.reset_hidden_states()

        # Create the input layer
        self.input_layer = nn.Linear(input_size, len(self.layers[0]))

        # Create forward hidden layers
        self.hidden_layers = nn.ModuleList([])
        for i in range(len(self.layers)-1):
            new_layer = nn.Linear(len(self.layers[i]),len(self.layers[i+1]))
            mask = self.adj_mat[np.ix_(self.layers[i],self.layers[i+1])] # connections from layer i to layer i+1
            prune.custom_from_mask(new_layer, name='weight', mask=torch.tensor(mask.T)) # delete fictive connections
            # prune.remove(new_layer, 'weight') # makes pruning permanent but removes mask -> trainable weights
            self.hidden_layers.append(new_layer)

        # Create the backward weights
        self.recurrent_layers = nn.ModuleList([]) # recurrent_layers[i](hidden_states) = layer j>i to i
        for i in range(len(self.layers)-1): # last layer should not have incomming reccurent connections
            # Find the backward connections to layer i
            nodes, nodes_layer = np.where(self.adj_mat[self.layers[i][-1]+1:,self.layers[i]]==1)
            
            new_layer = nn.Linear(self.n_neurons, len(self.layers[i]), bias=False) # no bias for backward connection
            mask = self.adj_mat[:,self.layers[i]]
            prune.custom_from_mask(new_layer, name='weight', mask=torch.tensor(mask.T)) # delete fictive connections
            # prune.remove(new_layer, 'weight') # makes pruning permanent but removes mask -> trainable weights
            self.recurrent_layers.append(new_layer)
        
        # create skip connections
        nb_nodes = len(self.layers[0])+len(self.layers[1]) # nb nodes in 2 first layers
        self.skip_layers = nn.ModuleList([]) # skip_layers[i](hidden_states) = layer j<i to hidden layer i+1
        for i in range(2,len(self.layers)): # first 2 layers should not have skip connections
            # Find the backward connections to layer i
            nodes, nodes_layer = np.where(self.adj_mat[:self.layers[i][0],self.layers[i]]==1)
            new_layer = nn.Linear(nb_nodes, len(self.layers[i]), bias=False) # no bias for skip connection
            mask = self.adj_mat[:nb_nodes,self.layers[i]]
            prune.custom_from_mask(new_layer, name='weight', mask=torch.tensor(mask.T)) # delete fictive connections
            # prune.remove(new_layer, 'weight') # makes pruning permanent but removes mask -> trainable weights
            self.skip_layers.append(new_layer)
            nb_nodes += len(self.layers[i])

        # Create the output layer
        self.output_layer = nn.Linear(len(self.layers[-1]), output_size)

    def forward(self, x):
        next_hidden_states = torch.empty(x.shape[0], self.n_neurons) if x.dim() > 1 else torch.empty(self.n_neurons)
        skips = [] # list of current states for skip connections

        # Input layer
        x = self.activation(self.input_layer(x) + self.recurrent_layers[0](self.hidden_states))
        next_hidden_states[...,self.layers[0]] = x
        skips.append(x)

        # Hidden layers
        for i in range(len(self.layers)-1):
            if i == 0: # no skip connection from layer 0 to 1
                x = self.hidden_layers[i](x) + self.recurrent_layers[i+1](self.hidden_states)
            elif i == len(self.layers)-2: # no recurrent connection for last hidden layer
                x = self.hidden_layers[i](x) + self.skip_layers[i-1](torch.concat(skips, dim=-1)) 
            else:
                x = (self.hidden_layers[i](x) 
                     + self.recurrent_layers[i+1](self.hidden_states)
                     + self.skip_layers[i-1](torch.concat(skips, dim=-1))
                )
            x = self.activation(x)
            next_hidden_states[...,self.layers[i+1]] = x
            skips.append(x)

        # Output layer
        x = self.output_layer(x) # no activation nor recurrent/skip connection for the last one

        self.hidden_states = next_hidden_states

        return x

    def reset_hidden_states(self, hidden_states=None, hidden_size=None, method='zero'):
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
                assert hidden_states.shape == (self.n_neurons, hidden_size), \
                    f'hidden_states should have shape {(self.n_neurons, hidden_size)}; received {hidden_states.shape}'
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