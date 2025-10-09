from functools import partial

from netam.dasm import DASMBurrito, DASMDataset

from dnsmex.dxsm_zoo import HPARAMS
from dnsmex import dxsm_zoo

validation_burrito_of_pcp_df = partial(
    dxsm_zoo.validation_burrito_of_pcp_df, DASMBurrito, DASMDataset
)
validation_burrito_of = partial(
    dxsm_zoo.validation_burrito_of, DASMBurrito, DASMDataset
)
train_model = partial(dxsm_zoo.train_model, DASMBurrito, DASMDataset)
update_model = partial(dxsm_zoo.update_model, DASMBurrito, DASMDataset)
retrain_model = partial(dxsm_zoo.retrain_model, DASMBurrito, DASMDataset)
create_model = partial(dxsm_zoo.create_model, DASMBurrito)
write_branch_lengths = partial(dxsm_zoo.write_branch_lengths, DASMBurrito, DASMDataset)

MODEL_NAMES = ["single"] + [f"dasm_{key}" for key in HPARAMS.keys()]
