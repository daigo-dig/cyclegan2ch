import numpy as np
from operator import itemgetter
datasetA = ["a","b","c","d"]
datasetB = [1,2,3,4]
data_size = len(datasetA)
sample_size = 2
shuffled_idx = np.random.choice(np.arange(data_size), sample_size, replace=False)
datasetA_random_pickup = itemgetter(*shuffled_idx)(datasetA)
datasetB_random_pickup = itemgetter(*shuffled_idx)(datasetB)
print(datasetA[1])
print(datasetA_random_pickup)
print(datasetB_random_pickup)


a = np.arange(128*128).reshape((128,128,1))
b = np.arange(128*128).reshape((128,128,1))
print(a.shape)
print(b.shape)
c = np.stack([a,b])
print(c.shape)
