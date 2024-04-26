import json
import numpy as np

from settings import *

mat = np.zeros((15,15))
layers = [[0,1,2,3,4],
          [5,6,7,8,9],
          [10,11,12,13,14]]

# forward connections
for i,layer in enumerate(layers[:-1]):
    targets = layers[i+1]
    for j in layer:
        for k in targets: # 4 direct connections
            if k+5!=j:
                mat[j,k] = 1

# backward connections
# layer 2 -> 1
mat[5,1] = 1
mat[7,[0,2]] = 1

# layer 3 -> 1
mat[12,3] = 1
mat[14,[1,3]] = 1

# layer 3-> 2
mat[10, 6] = 2
mat[[13,14], [7,9]] = 1

# skip connections
mat[1, 12] = 1
mat[[3,4], 11] = 1

json_mat = mat.tolist()

dico['medium'] = {
    'matrix': json_mat,
    'layers': layers
}
with open('adj_matrices.json', 'w') as file:
    dico = json.dump(dico, file)