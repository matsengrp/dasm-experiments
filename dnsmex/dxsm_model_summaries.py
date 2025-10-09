import pandas as pd

from netam.common import parameter_count_of_model


def abbreviate_number(num):
    """Abbreviate numbers for readability."""
    if num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.0f}K"
    else:
        return str(num)


def build_summary_df(zoo_module):
    model_summary_rows = []

    for model_name in zoo_module.MODEL_NAMES:
        if model_name == "single":
            model_summary_rows.append((1, model_name, 0, 0, 0, 1))
        else:
            model = zoo_module.create_model(model_name)
            parameter_count = parameter_count_of_model(model)
            abbrv_count = abbreviate_number(parameter_count)
            layer_count = model.hyperparameters["layer_count"]

            head_count = model.nhead
            model_summary_rows.append(
                (
                    abbrv_count,
                    model_name,
                    head_count,
                    model.d_model_per_head,
                    layer_count,
                    parameter_count,
                )
            )

    summary_df = pd.DataFrame(
        model_summary_rows,
        columns=[
            "abbrv_param_count",
            "model",
            "head_count",
            "d_model_per_head",
            "layer_count",
            "parameters",
        ],
    )
    return summary_df
