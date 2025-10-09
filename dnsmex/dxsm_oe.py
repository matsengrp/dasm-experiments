from copy import deepcopy
import os
from abc import ABC
import torch
from dnsmex.dxsm_zoo import validation_burrito_of_pcp_df
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

import numpy as np

from natsort import natsorted

from netam.sequences import (
    translate_sequences,
    heavy_light_mask_of_aa_idxs,
    AA_AMBIG_IDX,
)
from netam.common import clamp_probability
from dnsmex.dxsm_data import pcp_df_of_multiname, anarci_dict

from netam.oe_plot import (
    annotate_sites_df,
    get_site_substitutions_df,
    get_subs_and_preds_from_mutabilities_df,
    plot_observed_vs_expected,
    plot_sites_observed_vs_expected,
    get_numbering_dict,
    get_site_subs_acc_df,
    get_sub_acc_from_csp_df,
    plot_sites_subs_acc,
)


def chain_mask_func_of_chain(chain_type):
    if chain_type == "heavy":
        return lambda aa_idxs: heavy_light_mask_of_aa_idxs(aa_idxs)["heavy"]
    elif chain_type == "light":
        return lambda aa_idxs: heavy_light_mask_of_aa_idxs(aa_idxs)["light"]
    else:
        raise ValueError(f"Unknown chain type: {chain_type}")


def perplexity_of_probs(probs):
    """Calculate the perplexity of an array of probabilities.

    Args:
        probs (array-like): An array of probabilities. Values should be in
        (0, 1], but we clip them below at 1e-12 to avoid log(0).

    Returns:
        float: The perplexity of the input probabilities.
    """
    epsilon = 1e-12
    probs = np.clip(probs, epsilon, None)
    return np.exp(-np.mean(np.log(probs)))


def pcp_index_arr_of_pcp_df(pcp_df):
    """Build a numpy array of pcp indices for every site from a DataFrame of parent-
    child pairs which contains a 'parent_aa' column which is the translated parent
    sequence.

    That is, the return value will repeat the index of each row in the DataFrame
    'length' times, where 'length' is the length of the 'parent_aa' string.

    'length' is the number of sites before any separator token (OE plots may only be
    built on heavy chain sequences)
    """
    pcp_index_list = []

    for idx, row in pcp_df.iterrows():
        length = len(row["parent_aa"])
        pcp_index_list.extend([idx] * length)

    pcp_index_arr = np.array(pcp_index_list, dtype=int)
    return pcp_index_arr


def perplexities_of_plotter(plotter):
    """Calculate perplexities.

    * csp_perplexity: perplexity of the conditional substitution probabilities
        for sites where there has been a substitution. Note that these
        probabilities are calculated conditional on there being a substitution.
    * unsubstituted_perplexity: perplexity of the probability of the parent
        amino acid for sites without a substitution. This is analogous to the
        "germline residues" perplexity in the AbLang2 paper (Olson et al. 2024).
    """
    return {
        "csp_perplexity": perplexity_of_probs(
            # The oe_csp_df has been filtered to only include rows where there
            # was a substitution. The query gets us the rows corresponding to
            # the observed amino acid.
            plotter.oe_csp_df.query("is_target == True")["prob"]
        ),
        "unsubstituted_perplexity": perplexity_of_probs(
            # Here our query gets us the rows where there has not been a
            # substitution, and by taking 1 - prob we get the probability of not
            # having a substitution.
            1.0
            - plotter.oe_plot_df.query("mutation == False")["prob"]
        ),
    }


def sites_oe_plots_of_plotter_dict(plotter_dict, fig=None):
    plotter_count = len(plotter_dict)
    if fig is None:
        fig, axs = plt.subplots(
            2 * plotter_count, 1, figsize=(15, 3.5 * 2 * plotter_count)
        )
    else:
        axs = fig.get_axes()

    results_dfs = []

    for ax_idx, (v_family, plotter) in enumerate(plotter_dict.items()):
        sites_ax = axs[2 * ax_idx]
        results = plotter.sites_oe_plot(sites_ax)
        sites_ax.set_xlabel("")
        csp_ax = axs[2 * ax_idx + 1]

        csp_results = plotter.csp_oe_plot(None, csp_ax)
        results["total_subacc"] = csp_results["total_subacc"]
        results.update(perplexities_of_plotter(plotter))

        if ax_idx == plotter_count - 1:
            csp_ax.set_xlabel(
                f"{plotter.numbering_type} numbering", fontsize=plotter.label_size
            )
        else:
            csp_ax.set_xlabel("")
        csp_ax.text(
            0.02,
            0.95,
            f'sub. acc.={results["total_subacc"]:.3g}\nCSP perp.={results["csp_perplexity"]:.3g}',
            verticalalignment="top",
            horizontalalignment="left",
            transform=csp_ax.transAxes,
            fontsize=15,
        )
        for site, subacc in zip(
            plotter.numbering["reference", 0], csp_results["site_subacc"]
        ):
            results[f"site_{site}"] = subacc
        results_df = pd.DataFrame(results, index=[0])
        results_df["model_nickname"] = plotter.crepe_basename
        results_df["data_description"] = plotter.dataset_name
        results_dfs.append(results_df)

    results_df = pd.concat(results_dfs)

    plt.tight_layout()

    return fig, results_df


class OEPlotter(ABC):
    def __init__(
        self,
        dataset_name,
        crepe_prefix,
        pcp_df,
        anarci_path,
        val_burrito,
        burrito_predictions,
        chain_type,
        min_log_prob=None,
    ):
        print(f"Preparing {chain_type} chain plot data")
        self.dataset_name = dataset_name
        self.crepe_basename = os.path.basename(crepe_prefix)

        self.chain_type = chain_type
        self.recompute_title_str()

        if "imgt" in anarci_path:
            self.numbering_type = "IMGT"
        elif "chothia" in anarci_path:
            raise ValueError(f"Get in touch with the authors to add Chothia numbering.")
            # Here are Kevin's notes on the Chothia numbering:
            # CDRH1: [26, 32]
            # CDRH2: [52A, 55]
            # CDRH3: [96, 101]
            # We would need to add those below.
            self.numbering_type = "Chothia"
        else:
            raise ValueError(f"Unknown numbering type in anarci_path: {anarci_path}")

        self.pcp_df = pcp_df
        self.numbering, _ = get_numbering_dict(
            anarci_path, self.pcp_df, verbose=True, checks="imgt"
        )
        self.burrito_predictions = burrito_predictions
        self.val_burrito = val_burrito

        # Add validation check for dataset lengths
        assert len(val_burrito.val_dataset) == len(pcp_df), (
            f"Validation dataset length ({len(val_burrito.val_dataset)}) "
            f"does not match pcp_df length ({len(pcp_df)})"
        )

        self._prepare_parent_for_oe_plot()

        self.oe_plot_df = self.oe_plot_df_of_burrito(
            val_burrito,
            self.pcp_df,
            self.chain_type,
            val_predictions=self.burrito_predictions,
        )

        if min_log_prob is not None:
            self.binning = np.linspace(min_log_prob, 0, 101)
        else:
            self.binning = None

        self._site_sub_probs_df = None
        self._mut_obs_pred_df = None
        self.label_size = 15

    def recompute_title_str(self):
        chain_str = "" if "IG" in self.dataset_name else f" {self.chain_type}"
        self.title_str = f"{self.crepe_basename} on {self.dataset_name}{chain_str}"
        return self.title_str

    @classmethod
    def val_predictions_of_burrito(cls, *args, **kwargs):
        raise NotImplementedError(
            f"val_predictions_of_burrito must be implemented on {cls}."
        )

    @classmethod
    def heavy_light_plotter_pair(
        cls,
        crepe_prefix,
        dataset_name,
        branch_length_path,
        anarci_paths,
        min_log_prob,
    ):
        """Produce a pair of OEPlotter objects for heavy and light chains.

        Args:
            crepe_prefix: The path to the crepe model.
            dataset_name: The name of the dataset.
            branch_length_path: The path to the branch length file.
            anarci_paths: A dictionary with keys "heavy"and "light" for heavy and light chain,
                and values of the ANARCI path.
            min_log_prob: The minimum log probability to use for binning.

        Returns:
            dict: A dictionary with keys "heavy"and "light" for heavy and light chain,
                and values of OEPlotter objects.
        """
        return cls.heavy_light_plotter_pair_of_pcp_df(
            crepe_prefix,
            pcp_df_of_multiname(dataset_name),
            dataset_name,
            branch_length_path,
            anarci_paths,
            min_log_prob,
        )

    @classmethod
    def heavy_light_plotter_pair_of_pcp_df(
        cls,
        crepe_prefix,
        pcp_df,
        dataset_name,
        branch_length_path,
        anarci_paths,
        min_log_prob,
    ):
        pcp_df = pcp_df.copy().reset_index(drop=True)
        print("Loading model")
        val_burrito = validation_burrito_of_pcp_df(
            cls.burrito_cls,
            cls.dataset_cls,
            crepe_prefix,
            pcp_df.copy(),
            None,
        )
        if branch_length_path is not None:
            print("Loading branch lengths")
            val_burrito.val_dataset.load_branch_lengths(branch_length_path)
        else:
            val_burrito.standardize_and_optimize_branch_lengths()

        predictions = cls.val_predictions_of_burrito(val_burrito)

        pcp_df_heavy = pcp_df
        pcp_df_light = pcp_df.copy()

        for colname in pcp_df.columns:
            shortened_colname = colname[: -len("_heavy")]
            if colname.endswith("_heavy"):
                pcp_df_heavy[shortened_colname] = pcp_df_heavy[colname]
                pcp_df_heavy.drop(columns=colname, inplace=True)
                pcp_df_light.drop(columns=colname, inplace=True)
            if colname.endswith("_light"):
                pcp_df_light[shortened_colname] = pcp_df_light[colname]
                pcp_df_light.drop(columns=colname, inplace=True)
                pcp_df_heavy.drop(columns=colname, inplace=True)

        result_dict = {}
        if pcp_df_heavy["parent"].str.len().max() > 0 and "heavy" in anarci_paths:
            result_dict["heavy"] = cls(
                dataset_name,
                crepe_prefix,
                pcp_df_heavy,
                anarci_paths["heavy"],
                val_burrito,
                predictions,
                "heavy",
                min_log_prob=min_log_prob,
            )
        # We can only look at light chain if the model accepts paired sequences
        if (
            pcp_df_light["parent"].str.len().max() > 0
            and "light" in anarci_paths
            and val_burrito.model.hyperparameters["known_token_count"]
            > AA_AMBIG_IDX + 1
        ):
            result_dict["light"] = cls(
                dataset_name,
                crepe_prefix,
                pcp_df_light,
                anarci_paths["light"],
                val_burrito,
                predictions,
                "light",
                min_log_prob=min_log_prob,
            )
        return result_dict

    @property
    def site_sub_probs_df(self):
        if self._site_sub_probs_df is None:
            print("Computing site substitution probabilities dataframe...")
            self._site_sub_probs_df = annotate_sites_df(
                self.oe_plot_df,
                self.pcp_df,
                self.numbering,
                add_codons_aas=True,
            )
            self._site_sub_probs_df = self._site_sub_probs_df.loc[
                natsorted(
                    self._site_sub_probs_df.index,
                    key=lambda x: self._site_sub_probs_df.loc[x, "site"],
                )
            ]
        return self._site_sub_probs_df

    @property
    def mut_obs_pred_df(self):
        if self._mut_obs_pred_df is None:
            print("Computing mutation probabilities dataframe...")
            self._muts_obs_pred_df = get_site_substitutions_df(
                get_subs_and_preds_from_mutabilities_df(self.oe_plot_df, self.pcp_df),
                self.numbering,
            )
        return self._mut_obs_pred_df

    def oe_plot(self, ax):
        results = plot_observed_vs_expected(
            self.oe_plot_df, None, ax, None, binning=self.binning
        )
        ax.set_title(self.title_str, fontsize=self.label_size)
        results.pop("counts_twinx_ax")
        return results

    def sites_oe_plot(self, ax):
        if len(self.oe_plot_df) == 0:
            # Plot "no substitutions observed" on axes
            ax.text(
                0.5,
                0.5,
                "No substitutions observed",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=self.label_size,
            )
            results = {"overlap": float("nan"), "residual": float("nan")}
        else:
            results = plot_sites_observed_vs_expected(
                self.site_sub_probs_df, ax, self.numbering
            )
            ax.legend(fontsize=self.label_size, loc="upper right")
            ax.text(
                0.02,
                0.95,
                f'overlap={results["overlap"]:.3g}\nresidual={results["residual"]:.3g}',
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax.transAxes,
                fontsize=self.label_size,
            )
        ax.set_title(self.title_str, fontsize=self.label_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=self.label_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=self.label_size)
        for idx, label in enumerate(ax.get_xticklabels()):
            if idx % 2 == 1:
                label.set_visible(False)
        ax.tick_params(axis="y", labelsize=self.label_size)
        return results

    def _prepare_parent_for_oe_plot(self):
        if not all(self.pcp_df["parent"].str.len() > 0):
            raise ValueError("Some parent sequences are empty. ")
        self.pcp_df["parent_aa"] = translate_sequences(self.pcp_df["parent"])

    @classmethod
    def write_sites_oe(
        cls,
        crepe_prefix,
        dataset_name,
        branch_length_path,
        csv_output_path,
        fig_out_path,
        min_log_prob=None,
        v_families=["IGHV1", "IGHV3", "IGHV4"],
        replace_title=False,
        fig=None,
    ):
        try:
            anarci_paths = anarci_dict[dataset_name]
        except KeyError:
            raise ValueError(
                f"Dataset name {dataset_name} isn't indexed in the ANARCI dict."
            )

        plotters = cls.heavy_light_plotter_pair(
            crepe_prefix=crepe_prefix,
            dataset_name=dataset_name,
            branch_length_path=branch_length_path,
            anarci_paths=anarci_paths,
            min_log_prob=min_log_prob,
        )
        v_families_dict = {
            "heavy": [family for family in v_families if "IGH" in family],
            "light": [
                family for family in v_families if "IGL" in family or "IGK" in family
            ],
        }
        plotter_dict = {}
        for chain_key, plotter in plotters.items():
            v_families = v_families_dict[chain_key]
            if len(v_families) == 0:
                plotter_dict["All " + chain_key + "V Families"] = plotter
            else:
                # Notice that values in `v_families_dict` are disjoint, so we
                # need not use chain_key to differentiate between heavy and light
                plotter_dict.update(
                    {
                        v_family: plotter.restrict_to_v_family(
                            v_family, replace_title=replace_title
                        )
                        for v_family in v_families
                    }
                )

        fig, results_df = cls.sites_oe_plots_of_plotter_dict(plotter_dict, fig=fig)
        fig.savefig(fig_out_path)
        results_df.to_csv(csv_output_path, index=False)

        return plotters, plotter_dict

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

        chain_mask_func = chain_mask_func_of_chain(chain_type)

        df_dict = {"prob": [], "mutation": []}
        for pcp_index, (row, val_prediction) in enumerate(
            zip(val_dataset, val_predictions)
        ):
            chain_mask = chain_mask_func(row["aa_children_idxs"])
            # Sequences are padded with ambiguous bases, so chain mask may have
            # more True values than the length of the parent sequence.
            parent_length = len(pcp_df.loc[pcp_index, "parent_aa"])
            ignore_mask = row["mask"][chain_mask]
            aa_subs_indicator = row["subs_indicator"]

            subs_pos_pred = clamp_probability(
                torch.sum(torch.exp(val_prediction), dim=-1)
            )
            subs_pos_pred = subs_pos_pred.detach().cpu().numpy()
            subs_pos_pred = subs_pos_pred[chain_mask]
            subs_pos_pred[~ignore_mask] = 0.0
            subs_pos_pred = subs_pos_pred[:parent_length]

            aa_subs_indicator = aa_subs_indicator[chain_mask]
            aa_subs_indicator[~ignore_mask] = False
            aa_subs_indicator = aa_subs_indicator[:parent_length].detach().cpu().numpy()
            df_dict["prob"].extend(subs_pos_pred)
            df_dict["mutation"].extend(aa_subs_indicator)

        oe_plot_df = pd.DataFrame(df_dict)
        oe_plot_df["mutation"] = oe_plot_df["mutation"].astype(bool)

        pcp_index_arr = pcp_index_arr_of_pcp_df(pcp_df)
        assert pcp_index_arr.shape[0] == len(oe_plot_df)
        oe_plot_df["pcp_index"] = pcp_index_arr
        return oe_plot_df

    @classmethod
    def oe_csp_df_of_burrito(cls, burrito, pcp_df, chain_type, val_predictions=None):
        """Makes a dataframe with information relevant for plotting the CSP accuracy.
        For each substitution there are 20 rows.

        Args:
            burrito (Burrito): a Burrito object.
            pcp_df: a dataframe with the columns "v_family", "parent", "parent_aa",
                "aa_parents_idxs", "aa_children_idxs".
            chain_type: A string, either ``heavy`` or ``light``.
            val_predictions: The predictions of the Burrito object, computed using ``dxsm_oe.val_predictions_of_burrito``. If None, they will be
                calculated.

        Output DF columns:
        * pcp_index (this is 0-based index of the parent-child pair in the dataset)
        * site = position of the substitution (0-based index in the sequence string)
        * aa = amino acid
        * prob = CSP for the amino acid
        * is_target = True/False whether the amino acid is the substitution result
        """

        rows = []
        chain_mask_func = chain_mask_func_of_chain(chain_type)
        if val_predictions is None:
            val_predictions = cls.val_predictions_of_burrito(burrito)

        csp_preds = torch.nn.functional.softmax(val_predictions, dim=-1)
        for pcp_index, (row, csp_pred) in enumerate(
            zip(burrito.val_dataset, csp_preds)
        ):
            subs_indicator = row["subs_indicator"]
            aa_child_idxs = row["aa_children_idxs"]
            chain_mask = chain_mask_func(aa_child_idxs)
            v_family = pcp_df.loc[pcp_index, "v_family"]
            # Loop over sites in the sequence.
            for site, (has_subs, probs, target_aa, consider_site) in enumerate(
                zip(
                    subs_indicator[chain_mask],
                    csp_pred[chain_mask],
                    aa_child_idxs[chain_mask],
                    row["mask"][chain_mask],
                )
            ):
                if has_subs:
                    # Loop over target amino acids.
                    for aa_idx, prob in enumerate(probs):
                        rows.append(
                            {
                                "pcp_index": pcp_index,
                                "site": site,
                                "aa": aa_idx,
                                "prob": prob.item() if consider_site else 0.0,
                                "is_target": aa_idx == target_aa.item(),
                                "v_family": v_family,
                            }
                        )

        oe_csp_df = pd.DataFrame(rows)
        oe_csp_df["is_target"] = oe_csp_df["is_target"].astype(bool)
        return oe_csp_df

    def csp_oe_plot(self, count_ax, sub_acc):
        """Make a plot of the CSP accuracy.

        Note that either of the axes can be None, in which case the corresponding plot
        will not be made.
        """
        subacc_df = get_site_subs_acc_df(
            get_sub_acc_from_csp_df(self.oe_csp_df, self.pcp_df, 1), self.numbering
        )
        if len(subacc_df) == 0:
            return {"total_subacc": 0, "site_subacc": []}
        results = plot_sites_subs_acc(subacc_df, count_ax, sub_acc, self.numbering)
        if count_ax is not None:
            count_ax.text(
                0.02,
                0.95,
                f'sub. acc.={results["total_subacc"]:.3g}',
                verticalalignment="top",
                horizontalalignment="left",
                transform=count_ax.transAxes,
                fontsize=15,
            )
            count_ax.xaxis.label.set_visible(False)
            count_ax.tick_params(axis="x", labelbottom=False)
            count_ax.set_ylabel("no. of substitutions", fontsize=20, labelpad=10)
            count_ax.legend(loc="upper left", fontsize=16)

        if sub_acc is not None:
            for cdr_bounds in [("27", "38"), ("56", "65"), ("105", "117")]:
                xlower = self.numbering[("reference", 0)].index(cdr_bounds[0])
                xupper = self.numbering[("reference", 0)].index(cdr_bounds[1])
                sub_acc.add_patch(
                    Rectangle(
                        (xlower - 0.5, 0),
                        xupper - xlower + 1,
                        sub_acc.get_ylim()[1],
                        color="#E69F00",
                        alpha=0.2,
                    )
                )

            sub_acc.set_xlabel("IMGT numbering", fontsize=20, labelpad=10)
            sub_acc.set_ylabel(f"sub. acc.", fontsize=20, labelpad=10)
            sub_acc.axhline(y=1 / 19, color="black", linestyle="--", linewidth=2)

        plt.tight_layout()
        return results

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

        # Create a new validation burrito with the subset dataset
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

        plotter.oe_csp_df = plotter.oe_csp_df[plotter.oe_csp_df["v_family"] == v_family]
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

        # Reset the derived dfs
        plotter._site_sub_probs_df = None
        plotter._mut_obs_pred_df = None

        return plotter

    @classmethod
    def sites_oe_plots_of_plotter_dict(cls, plotter_dict, fig=None):
        return sites_oe_plots_of_plotter_dict(plotter_dict, fig=fig)
