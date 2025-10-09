from functools import partial

from netam.ddsm import DDSMDataset
from dnsmex import dxsm_data

train_val_datasets_of_multiname = partial(
    dxsm_data.train_val_datasets_of_multiname, DDSMDataset
)

dataset_of_multiname = partial(dxsm_data.dataset_of_multiname, DDSMDataset)
