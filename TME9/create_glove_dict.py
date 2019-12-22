import pandas as pd
import pickle
from time import time

t0 = time()
df = pd.read_csv('~/datasets/glove/glove.6B.50d.txt', sep=" ", quoting=3, header=None, index_col=0)
glove = {key: val.values for key, val in df.T.items()}
print(time()-t0)

with open('/home/keyvan/datasets/glove/glove.6B.50d.pkl', 'wb') as fp:
    pickle.dump(glove, fp)

t0=time()
with open("/home/keyvan/datasets/glove/glove.6B.50d.pkl", "rb") as f:
    glove = pickle.load(f)
print(time()-t0)
print(next(iter(glove)))
