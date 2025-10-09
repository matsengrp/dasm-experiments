import os
import torch
import time
import random
from typing import Union

from netam.framework import load_crepe

from netam.common import pick_device, force_spawn

from dnsmex.dxsm_data import (
    dataset_of_pcp_df,
    pcp_df_of_multiname,
    train_val_datasets_of_multiname,
)
from netam.models import (
    TransformerBinarySelectionModelWiggleAct,
    SingleValueBinarySelectionModel,
    BidirectionalTransformerBinarySelectionModelWiggleAct,
    ParentIndependentBinarySelectionModel,
)

# Very helpful for debugging!
# torch.autograd.set_detect_anomaly(True)
# Helpful for detecting cyclic references to tensors that may cause cuda memory
# errors:
# from torch.utils.viz._cycles import warn_tensor_cycles
# warn_tensor_cycles()


def dxsm_pick_device(*args, **kwargs):
    """The MPS device isn't compatible with our masking scheme."""
    device = pick_device(*args, **kwargs)
    if device.type == "mps":
        print("Actually, using CPU")
        device = "cpu"
    return device


HPARAMS = {
    "sml": {
        "nhead": 4,
        "d_model_per_head": 4,
        "dim_feedforward": 1024,
        "layer_count": 3,
        "dropout_prob": 0.1,
    },
    "lrg": {
        "nhead": 8,
        "d_model_per_head": 4,
        "dim_feedforward": 4096,
        "layer_count": 5,
        "dropout_prob": 0.1,
    },
    "13k": {
        "nhead": 4,
        "d_model_per_head": 4,
        "dim_feedforward": 64,
        "layer_count": 3,
        "dropout_prob": 0.1,
    },
    "20k": {
        "nhead": 4,
        "d_model_per_head": 4,
        "dim_feedforward": 64,
        "layer_count": 5,
        "dropout_prob": 0.1,
    },
    "52k": {
        "nhead": 4,
        "d_model_per_head": 8,
        "dim_feedforward": 128,
        "layer_count": 3,
        "dropout_prob": 0.1,
    },
    "77k": {
        "nhead": 4,
        "d_model_per_head": 8,
        "dim_feedforward": 128,
        "layer_count": 5,
        "dropout_prob": 0.1,
    },
    "1m": {
        "nhead": 8,
        "d_model_per_head": 16,
        "dim_feedforward": 512,
        "layer_count": 5,
        "dropout_prob": 0.1,
    },
    # Doubling d_model_per_head to 32.
    "4m": {
        "nhead": 8,
        "d_model_per_head": 32,
        "dim_feedforward": 1024,
        "layer_count": 5,
        "dropout_prob": 0.1,
    },
    # Increasing the number of layers.
    "7m": {
        "nhead": 8,
        "d_model_per_head": 32,
        "dim_feedforward": 1024,
        "layer_count": 8,
        "dropout_prob": 0.1,
    },
    # Same shape as the smallest ESM model: esm2_t6_8M_UR50D.
    "8m": {
        "nhead": 20,
        "d_model_per_head": 16,
        "dim_feedforward": 1280,
        "layer_count": 6,
        # I do include dropout because our data volume is smaller than ESM's.
        "dropout_prob": 0.1,
    },
}

HPARAMS.update({f"bid_{k}": v for k, v in HPARAMS.items()})


def key_of_model_name(model_name):
    """Extract the key portion of a model name by removing known prefixes."""
    prefixes = ("dnsm_", "ddsm_", "dasm_")
    for prefix in prefixes:
        if model_name.startswith(prefix):
            return model_name.removeprefix(prefix)

    raise ValueError(f"Model name '{model_name}' must start with one of: {prefixes}")


def create_model(burrito_class, model_name, model_hparams=None):
    if burrito_class.model_type in ["ddsm", "dasm", "parent_independent"]:
        additional_hparams = {"output_dim": 20}
    else:
        additional_hparams = dict()
    additional_hparams.update({"model_type": burrito_class.model_type})
    if model_hparams is not None:
        additional_hparams.update(model_hparams)
    if model_name == "single":
        return SingleValueBinarySelectionModel(**additional_hparams)
    elif model_name == "parent_independent":
        return ParentIndependentBinarySelectionModel(**additional_hparams)
    else:
        try:
            hparams_key = key_of_model_name(model_name)
            hparams = HPARAMS[hparams_key]
        except KeyError:
            raise ValueError(f"Unknown model name: {model_name}")
        if burrito_class.model_type != model_name[: len(burrito_class.model_type)]:
            raise ValueError(
                f"Model name {model_name} does not match burrito class of {burrito_class.model_type}"
            )
        if hparams_key.startswith("bid_"):
            return BidirectionalTransformerBinarySelectionModelWiggleAct(
                **(hparams | additional_hparams)
            )
        else:
            return TransformerBinarySelectionModelWiggleAct(
                **(hparams | additional_hparams)
            )


def burrito_params_of(model_name):
    burrito_params = {
        "batch_size": 1024,
        "learning_rate": 0.001,
        "min_learning_rate": 1e-6,  # early stopping!
        "weight_decay": 1e-6,
    }

    if model_name is not None and model_name.endswith("wig_8m"):
        print("Halving batch size and learning rate for GPU memory reasons.")
        burrito_params["learning_rate"] /= 2
        burrito_params["batch_size"] //= 2

    return burrito_params


def trained_model_str(model_name, data_nickname, training_method):
    return f"{model_name}-{data_nickname}-{training_method}"


def trained_model_path(model_name, data_nickname, training_method):
    return f"trained_models/{trained_model_str(model_name, data_nickname, training_method)}"


def retrained_model_str(model_name, data_nickname, train_label, seed):
    return f"{trained_model_str(model_name, data_nickname, train_label)}-{seed}"


def retrained_model_path(model_name, data_nickname, train_label, seed):
    return f"trained_models/{retrained_model_str(model_name, data_nickname, train_label, seed)}"


# Generic functions that can get specialized to one case or another.


def validation_burrito_of_pcp_df(
    burrito_cls,
    dataset_cls,
    crepe_prefix,
    pcp_df,
    device,
):

    crepe = load_crepe(crepe_prefix, device=device)
    model = crepe.model
    model_known_token_count = model.hyperparameters["known_token_count"]
    neutral_model_name = model.hyperparameters["neutral_model_name"]
    multihit_model_name = model.hyperparameters["multihit_model_name"]

    dataset = dataset_of_pcp_df(
        dataset_cls,
        pcp_df,
        model_known_token_count,
        neutral_model_name,
        device=device,
        multihit_model_name=multihit_model_name,
    )
    burrito_params = burrito_params_of(None)
    return burrito_cls(
        None,
        dataset,
        model,
        **burrito_params,
    )


def validation_burrito_of(
    burrito_cls,
    dataset_cls,
    crepe_prefix,
    dataset_name,
    device,
):
    return validation_burrito_of_pcp_df(
        burrito_cls,
        dataset_cls,
        crepe_prefix,
        pcp_df_of_multiname(dataset_name, device=device),
        device,
    )


def write_branch_lengths(
    burrito_cls,
    dataset_cls,
    crepe_prefix,
    dataset_name,
    out_path,
):
    force_spawn()
    val_burrito = validation_burrito_of(
        burrito_cls,
        dataset_cls,
        crepe_prefix,
        dataset_name,
        None,
    )
    val_burrito.standardize_and_optimize_branch_lengths()
    val_burrito.val_dataset.export_branch_lengths(out_path)


def sleep_random_time():
    """Sleep a random amount of time, uniform up to 1 minute, to stagger GPU usage."""
    random.seed(os.getpid() + int(time.time() * 1e6))
    sleep_time = random.random() * 60
    print(
        f"Sleeping for {sleep_time:.2f} seconds before starting training so GPUs can start one at a time."
    )
    time.sleep(sleep_time)


def _train_with_model(
    burrito_cls: type,
    dataset_cls: type,
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    training_method: str,
    gpu_preference: Union[int, str],
    out_prefix: str = None,
    epochs: int = None,
    cycle_count: int = None,
    learning_rate: float = None,
) -> str:
    """Core training logic shared by train_model and update_model.

    Args:
        model_name (str): The name of the model.
        dataset_name (str): The nickname of the dataset.
        training_method (str): The training method to use: "fixed" or "joint".
        gpu_preference (int or str): See dxsm_pick_device for details.
        out_prefix (str): Optional output prefix (auto-determined if None)
        epochs (int): Number of epochs to train for
        cycle_count (int): Number of training cycles
        learning_rate (float): Learning rate for training

    Raises:
        ValueError: If an unknown training method is provided.
    """
    force_spawn()
    print(
        f"training {model_name} on {dataset_name} using {training_method}; gpu preference {gpu_preference}"
    )
    if dataset_name[:3] != "tst":
        sleep_random_time()
    device = dxsm_pick_device(gpu_preference)

    if out_prefix is None:
        out_prefix = trained_model_path(model_name, dataset_name, training_method)
    burrito_name = trained_model_str(model_name, dataset_name, training_method)

    model.to(device)
    model_known_token_count = model.hyperparameters["known_token_count"]
    neutral_model_name = model.hyperparameters["neutral_model_name"]
    multihit_model_name = model.hyperparameters["multihit_model_name"]

    _, train_dataset, val_dataset = train_val_datasets_of_multiname(
        dataset_cls,
        dataset_name,
        model_known_token_count,
        neutral_model_name,
        device=device,
        multihit_model_name=multihit_model_name,
    )

    burrito_params = burrito_params_of(model_name)

    # Override learning rate if provided
    if learning_rate is not None:
        burrito_params["learning_rate"] = learning_rate

    burrito = burrito_cls(
        train_dataset,
        val_dataset,
        model,
        **burrito_params,
        name=burrito_name,
    )
    optimize_bl_first_cycle = True
    if training_method == "joint":
        # If the single model has been trained, load those branch lengths.
        single_out_prefix = trained_model_path("single", dataset_name, training_method)
        if os.path.exists(single_out_prefix + ".train_branch_lengths.csv"):
            print(f"Loading branch lengths from {single_out_prefix}")
            burrito.load_branch_lengths(single_out_prefix)
            optimize_bl_first_cycle = False

    # Use provided values or defaults
    if epochs is None:
        epochs = 1000
        if dataset_name[:3] == "tst":
            epochs = 2

    if cycle_count is None:
        cycle_count = 4
        if model_name == "single":
            cycle_count = 2
        if dataset_name[:3] == "tst":
            cycle_count = 2
            optimize_bl_first_cycle = False
    if training_method == "fixed":
        single_out_prefix = trained_model_path("single", dataset_name, "joint")
        if os.path.exists(single_out_prefix + ".train_branch_lengths.csv"):
            print(f"Loading branch lengths from {single_out_prefix}")
            burrito.load_branch_lengths(single_out_prefix)
        else:
            raise ValueError(
                "No branch lengths found for fixed training. Refusing to fit."
            )
        burrito.simple_train(
            epochs=epochs,
            out_prefix=out_prefix,
        )
    elif training_method == "joint":
        burrito.joint_train(
            epochs=epochs,
            cycle_count=cycle_count,
            out_prefix=out_prefix,
            optimize_bl_first_cycle=optimize_bl_first_cycle,
        )
    else:
        raise ValueError(f"Unknown training method {training_method}")
    train_dataset.export_branch_lengths(out_prefix + ".train_branch_lengths.csv")
    val_dataset.export_branch_lengths(out_prefix + ".val_branch_lengths.csv")
    return out_prefix


def train_model(
    burrito_cls: type,
    dataset_cls: type,
    model_name: str,
    dataset_name: str,
    training_method: str,
    gpu_preference: Union[int, str],
):
    """Trains a model on a given dataset using a specified training method.

    Args:
        model_name (str): The name of the model.
        dataset_name (str): The nickname of the dataset.
        training_method (str): The training method to use: "fixed" or "joint".
        gpu_preference (int or str): See dxsm_pick_device for details.

    Raises:
        ValueError: If an unknown training method is provided.
    """
    model = create_model(burrito_cls, model_name)

    return _train_with_model(
        burrito_cls=burrito_cls,
        dataset_cls=dataset_cls,
        model=model,
        model_name=model_name,
        dataset_name=dataset_name,
        training_method=training_method,
        gpu_preference=gpu_preference,
    )


def update_model(
    burrito_cls: type,
    dataset_cls: type,
    crepe_prefix: str,
    dataset_name: str,
    gpu_preference: Union[int, str],
    out_prefix: str = None,
    epochs: int = 1000,
    cycle_count: int = 4,
    learning_rate: float = 0.0001,
) -> str:
    """Load a pre-trained model and continue training on additional dataset.

    Args:
        burrito_cls: The type of burrito to use for training
        dataset_cls: The type of dataset to use for training
        crepe_prefix: Path to existing trained model (without .yml/.pth extension)
        dataset_name: Dataset to train on (supports multiname format for mixed datasets)
        gpu_preference: GPU preference for training
        out_prefix: Optional output prefix (auto-determined if None)
        epochs: Number of epochs to train for (default: 1000)
        cycle_count: Number of training cycles for joint training (default: 4)
        learning_rate: Learning rate for fine-tuning (default: 0.0001, lower than normal 0.001)

    Returns:
        The output prefix for the updated model
    """
    # Load existing model
    device = dxsm_pick_device(gpu_preference)
    crepe = load_crepe(crepe_prefix, device=device)

    # Extract model name from crepe path
    # e.g., "dasm_77k-tst-joint" -> "dasm_77k"
    basename = os.path.basename(crepe_prefix)
    model_name = basename.split("-")[0]

    # Always use joint training for updates
    training_method = "joint"

    # Set default output prefix if not provided
    if out_prefix is None:
        out_prefix = f"{crepe_prefix}-UPDATED-{dataset_name}"

    # Use shared training logic with joint training
    return _train_with_model(
        burrito_cls=burrito_cls,
        dataset_cls=dataset_cls,
        model=crepe.model,
        model_name=model_name,
        dataset_name=dataset_name,
        training_method=training_method,
        gpu_preference=gpu_preference,
        out_prefix=out_prefix,
        epochs=epochs,
        cycle_count=cycle_count,
        learning_rate=learning_rate,
    )


def retrain_model(
    burrito_cls: type,
    dataset_cls: type,
    model_name: str,
    dataset_name: str,
    train_label: str,
    seed: int,
    burrito_param_overrides: Union[dict, str],
    epochs: int,
) -> str:
    """A simplified version of model training that doesn't do branch length
    optimization.

    Note that "retraining" is different than "resuming" training. The former
    starts from scratch (except using branch lengths from a previous train)
    while the latter continues from a previously saved crepe.

    Note that we use the seed to pick the GPU.

    Args:
        burrito_cls: The type of burrito to retrain
        dataset_cls: The type of dataset to use in retraining
        model: Model to retrain
        dataset_name: Dataset used for model retraining and validation
        train_label: Identifier for the retraining run
        seed: used for choosing the GPU
        burrito_param_overrides: A dictionary, or string representation of a
            dictionary, containing keyword arguments to pass to Burrito constructor
        epochs: Number of epochs to train for

    Returns:
        The output prefix for the retrained burrito.
    """
    torch.manual_seed(seed)
    if epochs >= 50:  # This is an actual training run, not a test.
        sleep_random_time()
    device = dxsm_pick_device(seed)

    model = create_model(burrito_cls, model_name)
    model_known_token_count = model.hyperparameters["known_token_count"]
    neutral_model_name = model.hyperparameters["neutral_model_name"]
    multihit_model_name = model.hyperparameters["multihit_model_name"]

    _, train_dataset, val_dataset = train_val_datasets_of_multiname(
        dataset_cls,
        dataset_name,
        model_known_token_count,
        neutral_model_name,
        device=device,
        multihit_model_name=multihit_model_name,
    )
    out_prefix = retrained_model_path(model_name, dataset_name, train_label, seed)
    burrito_name = retrained_model_str(model_name, dataset_name, train_label, seed)
    burrito_params = burrito_params_of(model_name)
    if isinstance(burrito_param_overrides, str):
        burrito_param_overrides = eval(burrito_param_overrides)
    burrito_params.update(burrito_param_overrides)
    burrito = burrito_cls(
        train_dataset,
        val_dataset,
        model.to(device),
        **burrito_params,
        name=burrito_name,
    )
    burrito.load_branch_lengths(
        "../"
        + burrito_cls.model_type
        + "-"
        + "train/"
        + trained_model_path(model_name, dataset_name, "joint")
    )
    burrito.reset_optimization()
    burrito.simple_train(epochs=epochs, out_prefix=out_prefix)
    return out_prefix
