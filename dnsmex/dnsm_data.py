from functools import partial

from netam.dnsm import DNSMDataset
from dnsmex import dxsm_data

train_val_datasets_of_multiname = partial(
    dxsm_data.train_val_datasets_of_multiname, DNSMDataset
)
dataset_of_multiname = partial(dxsm_data.dataset_of_multiname, DNSMDataset)
