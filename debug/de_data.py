import zarr
import numpy as np


# input_name = 'data/push_demo.zarr'
input_name = 'data/pusht/pusht_cchi_v7_replay.zarr'
# dataset_name = 'volumes/raw'
f = zarr.open(input_name)
# raw = f[dataset_name ]
raw = f
print(raw['meta']['episode_ends'])
raw_data = raw['meta']['episode_ends'][:]
print(raw_data)
