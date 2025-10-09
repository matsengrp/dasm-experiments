from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from netam.dnsm import DNSMDataset, DNSMBurrito
from dnsmex.dxsm_oe import chain_mask_func_of_chain
from dnsmex import dxsm_oe


def sites_oe_plots_of_plotter_dict(plotter_dict, fig=None):
    plotter_count = len(plotter_dict)
    if fig is None:
        fig, axs = plt.subplots(plotter_count, 1, figsize=(15, 3.5 * plotter_count))
        if plotter_count == 1:
            axs = [axs]
    else:
        axs = fig.get_axes()

    results_dfs = []

    for ax_idx, (v_family, plotter) in enumerate(plotter_dict.items()):
        ax = axs[ax_idx]
        results = plotter.sites_oe_plot(ax)
        if ax_idx == plotter_count - 1:
            ax.set_xlabel(
                f"{plotter.numbering_type} numbering", fontsize=plotter.label_size
            )
        else:
            ax.set_xlabel("")
        results_df = pd.DataFrame(results, index=[0])
        results_df["model_nickname"] = plotter.crepe_basename
        results_df["data_description"] = plotter.dataset_name
        results_dfs.append(results_df)

    results_df = pd.concat(results_dfs)

    plt.tight_layout()

    return fig, results_df


class OEPlotter(dxsm_oe.OEPlotter):
    burrito_cls = DNSMBurrito
    dataset_cls = DNSMDataset

    def csp_oe_plot(self, *args, **kwargs):
        raise NotImplementedError("CSP OE plots are not implemented for DNSM.")

    @classmethod
    def oe_plot_df_of_burrito(cls, burrito, pcp_df, chain_type, val_predictions=None):
        """Makes a dataframe with columns "prob" and "mutation" from the burrito.

        Args:
            burrito: a Burrito object for making predictions
            pcp_df: The pcp_df used to make the burrito must be provided to compute sequence lengths.
            chain_type: A string, either ``heavy`` or ``light``.
        """
        val_dataset = burrito.val_dataset

        if val_predictions is None:
            val_predictions = cls.val_predictions_of_burrito(burrito)

        val_predictions, val_selection_factors = val_predictions

        chain_mask_func = chain_mask_func_of_chain(chain_type)

        df_dict = {
            "neutral_prob": [],
            "selection_factor": [],
            "prob": [],
            "mutation": [],
        }
        for pcp_index, (row, val_prediction, val_selection) in enumerate(
            zip(val_dataset, val_predictions, val_selection_factors)
        ):
            chain_mask = chain_mask_func(row["aa_parents_idxs"])

            # Sequences are padded with ambiguous bases, so chain mask may have
            # more True values than the length of the parent sequence.
            parent_length = len(pcp_df.loc[pcp_index, "parent_aa"])
            ignore_mask = row["mask"][chain_mask]

            neutral_aa_mut_probs = row["log_neutral_aa_mut_probs"][chain_mask].exp()
            neutral_aa_mut_probs[~ignore_mask] = 0.0
            neutral_aa_mut_probs = neutral_aa_mut_probs[:parent_length]
            df_dict["neutral_prob"].extend(neutral_aa_mut_probs)

            selection_factors = val_selection[chain_mask].exp()
            selection_factors[~ignore_mask] = 1.0
            df_dict["selection_factor"].extend(selection_factors[:parent_length])

            predictions = val_prediction[chain_mask].clone()
            predictions[~ignore_mask] = 0.0
            df_dict["prob"].extend(predictions[:parent_length])

            aa_subs_indicator = row["aa_subs_indicator"]
            aa_subs_indicator = aa_subs_indicator[chain_mask][:parent_length]
            aa_subs_indicator = aa_subs_indicator.cpu().numpy()
            df_dict["mutation"].extend(aa_subs_indicator)

        oe_plot_df = pd.DataFrame(df_dict)
        oe_plot_df["mutation"] = oe_plot_df["mutation"].astype(bool)

        pcp_index_arr = dxsm_oe.pcp_index_arr_of_pcp_df(pcp_df)
        assert pcp_index_arr.shape[0] == len(oe_plot_df)
        oe_plot_df["pcp_index"] = pcp_index_arr
        return oe_plot_df

    def restrict_to_v_family(self, v_family, replace_title=True):
        """Make a new OEPlotter object which only considers sequences using the
        specified V gene."""
        plotter = deepcopy(self)
        plotter.dataset_name += f" {v_family}"
        if replace_title:
            plotter.title_str = v_family
        else:
            plotter.recompute_title_str()

        plotter.pcp_df = plotter.pcp_df[plotter.pcp_df["v_family"] == v_family]
        ingroup_pcp_indices = set(plotter.pcp_df.index)
        plotter.oe_plot_df = plotter.oe_plot_df[
            plotter.oe_plot_df["pcp_index"].isin(ingroup_pcp_indices)
        ]

        # Reset the derived dfs.
        plotter._site_sub_probs_df = None
        plotter._mut_obs_pred_df = None

        return plotter

    def restrict_to_v_family(self, v_family, replace_title=True):
        """Make a new OEPlotter object which only considers sequences using the
        specified V gene."""
        plotter = deepcopy(self)
        plotter.dataset_name += f"_{v_family}"
        if replace_title:
            plotter.title_str = v_family
        else:
            plotter.title_str += f"_{v_family}"

        plotter.pcp_df = plotter.pcp_df[plotter.pcp_df["v_family"] == v_family]

        # Create a new vaildation burrito with the subset dataset
        ingroup_pcp_indices = list(plotter.pcp_df.index)
        subset_val_dataset = plotter.val_burrito.val_dataset.subset_via_indices(
            ingroup_pcp_indices
        )
        new_val_burrito = deepcopy(plotter.val_burrito)
        new_val_burrito.val_dataset = subset_val_dataset
        plotter.val_burrito = new_val_burrito
        plotter.burrito_predictions = self.__class__.val_predictions_of_burrito(
            plotter.val_burrito
        )

        plotter.oe_plot_df = plotter.oe_plot_df[
            plotter.oe_plot_df["pcp_index"].isin(ingroup_pcp_indices)
        ]

        # Add validation check for dataset lengths after subsetting
        pcp_df = plotter.pcp_df
        val_burrito = plotter.val_burrito
        assert len(val_burrito.val_dataset) == len(pcp_df), (
            f"Validation dataset length ({len(val_burrito.val_dataset)}) "
            f"does not match pcp_df length ({len(pcp_df)})"
        )

        # Reset the derived dfs.
        plotter._site_sub_probs_df = None
        plotter._mut_obs_pred_df = None

        return plotter

    @classmethod
    def sites_oe_plots_of_plotter_dict(cls, plotter_dict, fig=None):
        return sites_oe_plots_of_plotter_dict(plotter_dict, fig=fig)

    @classmethod
    def val_predictions_of_burrito(cls, burrito):
        burrito.model.eval()
        val_loader = burrito.build_val_loader()
        predictions_list = []
        log_selection_factors_list = []
        for batch in tqdm(val_loader, desc="Calculating model predictions"):
            log_neutral_aa_mut_probs, log_selection_factors = (
                burrito.prediction_pair_of_batch(batch)
            )
            predictions = burrito.predictions_of_pair(
                log_neutral_aa_mut_probs, log_selection_factors
            )
            predictions_list.append(predictions.detach().cpu())
            log_selection_factors_list.append(log_selection_factors.detach().cpu())
        predictions = torch.cat(predictions_list, axis=0)
        log_selection_factors = torch.cat(log_selection_factors_list, axis=0)
        return predictions, log_selection_factors


val_predictions_of_burrito = OEPlotter.val_predictions_of_burrito
oe_plot_df_of_burrito = OEPlotter.oe_plot_df_of_burrito
oe_csp_df_of_burrito = OEPlotter.oe_csp_df_of_burrito
write_sites_oe = OEPlotter.write_sites_oe
