from functools import partial

from netam.dasm import DASMDataset
from dnsmex import dxsm_data

train_val_datasets_of_multiname = partial(
    dxsm_data.train_val_datasets_of_multiname, DASMDataset
)

dataset_of_multiname = partial(dxsm_data.dataset_of_multiname, DASMDataset)
