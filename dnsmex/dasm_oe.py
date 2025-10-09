import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

from netam.ddsm import zap_predictions_along_diagonal
from netam.dasm import DASMBurrito, DASMDataset
from netam.codon_table import CODON_AA_INDICATOR_MATRIX
from netam.sequences import AA_STR_SORTED
from dnsmex import dxsm_oe
from dnsmex.dxsm_oe import chain_mask_func_of_chain, pcp_index_arr_of_pcp_df
from netam.oe_plot import annotate_sites_df
from natsort import natsorted


def zapped_aa_log_preds_of(burrito, batch):
    """Get the log predictions for amino acid substitutions, zapped along the
    diagonal."""
    log_preds = burrito.predictions_of_batch(batch)
    aa_log_preds = torch.log((torch.exp(log_preds) @ CODON_AA_INDICATOR_MATRIX))
    aa_log_preds = zap_predictions_along_diagonal(
        aa_log_preds, batch["aa_parents_idxs"]
    )
    return aa_log_preds


class OEPlotter(dxsm_oe.OEPlotter):
    burrito_cls = DASMBurrito
    dataset_cls = DASMDataset

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
        super().__init__(
            dataset_name,
            crepe_prefix,
            pcp_df,
            anarci_path,
            val_burrito,
            burrito_predictions,
            chain_type,
            min_log_prob=min_log_prob,
        )
        print("Calculating oe csp dataframe")
        self.oe_csp_df = self.oe_csp_df_of_burrito(
            val_burrito,
            self.pcp_df,
            self.chain_type,
            val_predictions=self.burrito_predictions,
        )

    @classmethod
    def val_predictions_of_burrito(cls, burrito):
        burrito.model.eval()
        val_loader = burrito.build_val_loader()
        predictions_list = []
        for batch in tqdm(val_loader, desc="Calculating model predictions"):
            predictions = zapped_aa_log_preds_of(burrito, batch)
            predictions_list.append(predictions.detach().cpu())
        return torch.cat(predictions_list, axis=0)

    @property
    def aa_site_subs_selection_df(self):
        """
        Calculates site-specific amino acid substitution probabilities and selection factors.
        This method processes validation data to:
        1. Compute log selection factors for each site to every possible amino acid
        2. Assess if a mutation occured in each site in parent-child pair
        3. Annotate sites with additional metadata

        Returns:
        --------
        pd.DataFrame
            A DataFrame with expanded rows, each representing a potential amino acid substitution
            with its corresponding selection factor. PCP mutation status and CDR status are also
            included and are the same across all rows for a given site+pcp.
        Key Columns:
        - pcp_index: Unique identifier for the parent sequence
        - site: Genomic site of the substitution
        - selection_factor: DASM selection factor for the possible amino acid substitution
        - selection_factor_target_aa: Target amino acid for the selection factor
        - mutation: Boolean indicating if a substitution occurred at this site in parent-child pair; unrelated to selection_factor_target_aa
        - is_cdr: Whether the site is in a CDR region
        - parent_codon/aa: Original amino acid and codon
        - child_codon/aa: Substituted amino acid and codon
        """
        # Early return if result is cached
        if (
            hasattr(self, "_aa_site_subs_selection_df")
            and self._aa_site_subs_selection_df is not None
        ):
            return self._aa_site_subs_selection_df

        burrito = self.val_burrito
        pcp_df = self.pcp_df
        chain_type = self.chain_type
        numbering = self.numbering

        burrito.model.eval()
        val_loader = burrito.build_val_loader()
        log_selection_factors_list = []

        for batch in tqdm(val_loader, desc="Calculating DASM selection factors"):
            # Compute selection factors
            log_neutral_codon_probs, log_selection_factors = (
                burrito.prediction_pair_of_batch(batch)
            )
            log_selection_factors_list.append(log_selection_factors.detach().cpu())

        log_selection_factors_list = torch.cat(log_selection_factors_list, axis=0)
        chain_mask_func = chain_mask_func_of_chain(chain_type)

        df_dict = {
            "selection_factor": [],
            "mutation": [],
            "pcp_index": [],
        }

        pcp_indices = pcp_df.index.tolist()

        # Ensure we have the same number of rows in all datasets
        assert (
            len(pcp_indices)
            == len(burrito.val_dataset)
            == len(log_selection_factors_list)
        ), (
            f"Mismatch in dataset lengths: pcp_df={len(pcp_indices)}, "
            f"val_dataset={len(burrito.val_dataset)}, "
            f"log_selection_factors={len(log_selection_factors_list)}"
        )

        for pcp_idx, row, log_selection in zip(
            pcp_indices, burrito.val_dataset, log_selection_factors_list
        ):
            chain_mask = chain_mask_func(row["aa_children_idxs"])
            parent_length = len(pcp_df.loc[pcp_idx, "parent_aa"])
            ignore_mask = row["mask"][chain_mask]

            aa_subs_indicator = row["subs_indicator"]
            aa_subs_indicator = aa_subs_indicator[chain_mask]
            aa_subs_indicator[~ignore_mask] = False
            aa_subs_indicator = aa_subs_indicator[:parent_length].detach().cpu().numpy()
            df_dict["mutation"].extend(aa_subs_indicator)

            df_dict["pcp_index"].extend([pcp_idx] * len(aa_subs_indicator))

            selection_factors = log_selection[chain_mask].exp().detach().cpu().numpy()
            selection_factors[~ignore_mask] = 1.0
            df_dict["selection_factor"].extend(selection_factors[:parent_length])

        aa_site_subs_selection_df = pd.DataFrame(df_dict)
        aa_site_subs_selection_df["mutation"] = aa_site_subs_selection_df[
            "mutation"
        ].astype(bool)

        self._aa_site_subs_selection_df = annotate_sites_df(
            aa_site_subs_selection_df, pcp_df, numbering, add_codons_aas=True
        )
        self._aa_site_subs_selection_df = self._aa_site_subs_selection_df.loc[
            natsorted(
                self._aa_site_subs_selection_df.index,
                key=lambda x: self._aa_site_subs_selection_df.loc[x, "site"],
            )
        ]
        # Expand the DataFrame to include individual amino acid substitutions
        expanded_data = {col: [] for col in self._aa_site_subs_selection_df.columns}
        expanded_data["selection_factor_target_aa"] = []
        total_rows = len(self._aa_site_subs_selection_df)
        print(
            f"Expanding {total_rows} rows into {total_rows * len(AA_STR_SORTED)} rows..."
        )
        batch_size = 1000
        for batch_start in tqdm(
            range(0, total_rows, batch_size), desc="Expanding rows"
        ):
            batch_end = min(batch_start + batch_size, total_rows)
            batch = self._aa_site_subs_selection_df.iloc[batch_start:batch_end]
            for _, row in batch.iterrows():
                selection_factor_tensor = row["selection_factor"]
                assert len(selection_factor_tensor) == len(AA_STR_SORTED)
                for i, (selection_factor, aa) in enumerate(
                    zip(selection_factor_tensor, AA_STR_SORTED)
                ):
                    for col in self._aa_site_subs_selection_df.columns:
                        if col != "selection_factor":
                            expanded_data[col].append(row[col])
                    expanded_data["selection_factor"].append(selection_factor)
                    expanded_data["selection_factor_target_aa"].append(aa)
        print("Creating expanded DataFrame...")
        self._aa_site_subs_selection_df = pd.DataFrame(expanded_data)
        # Mask selection factors for identical parent-child amino acids
        mask = (
            self._aa_site_subs_selection_df.parent_aa
            == self._aa_site_subs_selection_df.selection_factor_target_aa
        )
        self._aa_site_subs_selection_df.loc[mask, "selection_factor"] = np.nan
        # Reorder columns for clarity
        column_order = [
            "pcp_index",
            "site",
            "selection_factor_target_aa",
            "selection_factor",
            "mutation",
            "is_cdr",
            "parent_codon",
            "parent_aa",
            "child_codon",
            "child_aa",
        ]
        self._aa_site_subs_selection_df = self._aa_site_subs_selection_df[column_order]
        return self._aa_site_subs_selection_df

    def restrict_to_v_family(self, *args, **kwargs):
        plotter = super().restrict_to_v_family(*args, **kwargs)
        # Reset the derived dfs.
        plotter._aa_site_subs_selection_df = None

        return plotter


oe_plot_df_of_burrito = OEPlotter.oe_plot_df_of_burrito
oe_csp_df_of_burrito = OEPlotter.oe_csp_df_of_burrito
write_sites_oe = OEPlotter.write_sites_oe
