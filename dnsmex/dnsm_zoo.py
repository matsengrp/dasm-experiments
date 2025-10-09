from functools import partial
import fire

from dnsmex.sync import sync
from dnsmex.dxsm_zoo import HPARAMS

from dnsmex import dxsm_zoo

from netam.dnsm import (
    DNSMBurrito,
    DNSMDataset,
)

validation_burrito_of_pcp_df = partial(
    dxsm_zoo.validation_burrito_of_pcp_df, DNSMBurrito, DNSMDataset
)

validation_burrito_of = partial(
    dxsm_zoo.validation_burrito_of, DNSMBurrito, DNSMDataset
)
train_model = partial(dxsm_zoo.train_model, DNSMBurrito, DNSMDataset)
update_model = partial(dxsm_zoo.update_model, DNSMBurrito, DNSMDataset)
retrain_model = partial(dxsm_zoo.retrain_model, DNSMBurrito, DNSMDataset)
create_model = partial(dxsm_zoo.create_model, DNSMBurrito)
write_branch_lengths = partial(dxsm_zoo.write_branch_lengths, DNSMBurrito, DNSMDataset)

MODEL_NAMES = ["single"] + [f"dnsm_{key}" for key in HPARAMS.keys()]


def our_sync():
    source_paths = """
    quokka:/home/matsen/re/dnsm-experiments-1/dnsm-train/trained_models
    quokka:/home/matsen/re/dnsm-experiments-1/dnsm-train/_logs
    quokka:/home/matsen/re/dnsm-experiments-1/dnsm-train/output
    """
    destination = "."
    sync(source_paths, destination)


def main():
    fire.Fire(
        {
            "train": train_model,
            "retrain": retrain_model,
            "sync": our_sync,
        }
    )
