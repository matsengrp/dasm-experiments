import torch
from tqdm import tqdm

from netam.ddsm import zap_predictions_along_diagonal, DDSMBurrito, DDSMDataset
from dnsmex import dxsm_oe


class OEPlotter(dxsm_oe.OEPlotter):
    burrito_cls = DDSMBurrito
    dataset_cls = DDSMDataset

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
        """Return log predicted probabilities for target aa's at each site, including
        selection effects."""
        burrito.model.eval()
        val_loader = burrito.build_val_loader()
        predictions_list = []
        for batch in tqdm(val_loader, desc="Calculating model predictions"):
            predictions = zap_predictions_along_diagonal(
                burrito.predictions_of_batch(batch), batch["aa_parents_idxs"]
            )
            predictions_list.append(predictions.detach().cpu())
        return torch.cat(predictions_list, axis=0)


oe_plot_df_of_burrito = OEPlotter.oe_plot_df_of_burrito
oe_csp_df_of_burrito = OEPlotter.oe_csp_df_of_burrito
write_sites_oe = OEPlotter.write_sites_oe
