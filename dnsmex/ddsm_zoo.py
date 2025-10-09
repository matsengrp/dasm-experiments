from functools import partial

from netam.ddsm import DDSMBurrito, DDSMDataset

from dnsmex.dxsm_zoo import HPARAMS
from dnsmex import dxsm_zoo

validation_burrito_of_pcp_df = partial(
    dxsm_zoo.validation_burrito_of_pcp_df, DDSMBurrito, DDSMDataset
)
validation_burrito_of = partial(
    dxsm_zoo.validation_burrito_of, DDSMBurrito, DDSMDataset
)
train_model = partial(dxsm_zoo.train_model, DDSMBurrito, DDSMDataset)
retrain_model = partial(dxsm_zoo.retrain_model, DDSMBurrito, DDSMDataset)
create_model = partial(dxsm_zoo.create_model, DDSMBurrito)
write_branch_lengths = partial(dxsm_zoo.write_branch_lengths, DDSMBurrito, DDSMDataset)

MODEL_NAMES = ["single"] + [f"ddsm_{key}" for key in HPARAMS.keys()]
