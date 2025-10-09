import random
import pandas as pd
from warnings import warn
from tqdm import tqdm

from netam.framework import load_pcp_df, add_shm_model_outputs_to_pcp_df
from netam.models import DEFAULT_NEUTRAL_MODEL
from netam.sequences import assert_pcp_valid
from netam import pretrained
from netam.common import parallel_df_apply
from dnsmex.local import localify

# Provides pd.DataFrame.progress_apply:
tqdm.pandas()


def validate_dataset_name_for_filesystem_safety(dataset_name: str):
    """Validate that dataset name is safe for use in filenames and paths.

    Dataset names should be simple nicknames without path separators or special
    characters that could cause issues in filenames or cross-platform compatibility.

    Args:
        dataset_name: The dataset name to validate

    Raises:
        ValueError: If the dataset name contains problematic characters
    """
    problematic_chars = ["/", "\\", ":", "*", "?", "<", ">", "|", '"', "'"]
    for char in problematic_chars:
        if char in dataset_name:
            raise ValueError(
                f"Dataset name '{dataset_name}' contains problematic character '{char}'. "
                f"Dataset names should be simple nicknames without path separators or special characters."
            )


dataset_dict = {
    "tst": "DATA_DIR/v3/v3convert_wyatt-10x-1p5m_fs-all_pcp_2024-04-29_NI_noN_no-naive_first100.csv.gz",
    "tstWithN": "DATA_DIR/v3/v3convert_tang-deepshm-prod-InclMutInv_pcp_2024-10-29_MASKED_NI_no-naive_sampled100WithNs_DXSMVALID.csv.gz",
    "tstPaired": "DATA_DIR/v3/v3convert_wyatt-10x-1p5m_paired-merged_fs-all_pcp_2024-11-21_no-naive_sample100AllMut.csv.gz",
    "v1jaffe": "DATA_DIR/v3/wyatt-10x-1p5m_fs-all-NoWinCheck_igh_pcp_2024-10-29_NI_noN_no-naive.csv.gz",
    "v1jaffeCC": "DATA_DIR/v3/wyatt-10x-1p5m_fs-all-NoWinCheck_igh_pcp_2024-10-29_NI_ConsCys_no-naive.csv.gz",
    "v1jaffeCC50k": "DATA_DIR/v3/wyatt-10x-1p5m_fs-all-NoWinCheck_igh_pcp_2024-10-29_NI_ConsCys_no-naive_downsample_50k.csv.gz",
    "v1jaffe50k": "DATA_DIR/v3/wyatt-10x-1p5m_fs-all-NoWinCheck_igh_pcp_2024-10-29_NI_noN_no-naive_downsample_50k.csv.gz",
    "v1jaffePaired": "DATA_DIR/v3/wyatt-10x-1p5m_fs-all-NoWinCheck-UnmutInv-GTR-paired-merged_pcp_2024-11-21_DXSMVALID_no-naive.csv.gz",
    "v1jaffePairedCC": "DATA_DIR/v3/wyatt-10x-1p5m_fs-all-NoWinCheck-UnmutInv-GTR-paired-merged_pcp_2024-11-21_DXSMVALID_no-naive_ConsCys_HL.csv.gz",
    "v2jaffePaired": "DATA_DIR/v3/wyatt-10x-1p5m_fs-all-UnmutInv_paired-merged_pcp_2024-11-22_no-naive_NI.csv.gz",
    "v2jaffePairedCC": "DATA_DIR/v3/wyatt-10x-1p5m_fs-all-UnmutInv_paired-merged_pcp_2024-11-22_no-naive_NI_ConsCys_HL.csv.gz",
    "v1tang": "DATA_DIR/v3/tang-deepshm-prod-NoWinCheck_igh_pcp_2024-10-29_MASKED_NI_noN_no-naive.csv.gz",
    "v1tangCC": "DATA_DIR/v3/tang-deepshm-prod-NoWinCheck_igh_pcp_2024-10-29_MASKED_NI_ConsCys_no-naive_DXSMVALID.csv.gz",
    "v1tangCC50k": "DATA_DIR/v3/tang-deepshm-prod-NoWinCheck_igh_pcp_2024-10-29_MASKED_NI_ConsCys_no-naive_DXSMVALID_downsample_50k.csv.gz",
    "v1tangWithN": "DATA_DIR/v3/tang-deepshm-prod-NoWinCheck_igh_pcp_2024-10-29_MASKED_NI_no-naive_DXSMVALID.csv.gz",
    "v1tangWithNaive": "DATA_DIR/v3/tang-deepshm-prod-NoWinCheck_igh_pcp_2024-10-29_MASKED_NI_DXSMVALID.csv.gz",
    "v1tangSingletonsCC": "DATA_DIR/v3/tang-deepshm-prod-NoWinCheck_CF1_igh_pcp_2025-02-15_MASKED_NI_ConsCys.csv.gz",
    "v1tangSingletons": "DATA_DIR/v3/tang-deepshm-prod-NoWinCheck_CF1_igh_pcp_2025-02-15_MASKED_NI_DXSMVALID.csv.gz",
    "v1tang50k": "DATA_DIR/v3/tang-deepshm-prod-NoWinCheck_igh_pcp_2024-10-29_MASKED_NI_noN_no-naive_downsample_50k.csv.gz",
    "v1rodriguezPrimersWithN": "DATA_DIR/v3/rodriguez-airr-seq-primer-prod-NoWinCheck-UnmutInv_igh_pcp_2024-04-01_MASKED_NI_no-naive_DXSMVALID.csv.gz",
    "v1rodriguez": "DATA_DIR/v3/rodriguez-airr-seq-race-prod-NoWinCheck_igh_pcp_2024-11-12_MASKED_NI_noN_no-naive.csv.gz",
    "v1rodriguezCC": "DATA_DIR/v3/rodriguez-airr-seq-race-prod-NoWinCheck_igh_pcp_2024-11-12_MASKED_NI_ConsCys_no-naive_DXSMVALID.csv.gz",
    "v1rodriguezWithN": "DATA_DIR/v3/rodriguez-airr-seq-race-prod-NoWinCheck_igh_pcp_2024-11-12_MASKED_NI_no-naive_DXSMVALID.csv.gz",
    "v1flairr": "DATA_DIR/v3/ford-flairr-seq-prod-NoWinCheck-UnmutInv_igh_pcp_2024-04-01_MASKED_NI_noN_no-naive.csv.gz",
    "v1vanwinkleigkTest": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_test_DXSMVALID_no-naive.csv.gz",
    "v1vanwinkleigkTestCC": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_test_DXSMVALID_ConsCys_no-naive.csv.gz",
    "v1vanwinkleigkTrain": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_no-naive.csv.gz",
    "v1vanwinkleigkTrainCC": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive.csv.gz",
    "v1vanwinkleigkTrainCC50k": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_50k.csv.gz",
    "v1vanwinkleigkTrainCC100k": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_100k.csv.gz",
    "v1vanwinkleigkTrainCC250k": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_250k.csv.gz",
    "v1vanwinkleigkTrainCC500k": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_500k.csv.gz",
    "v1vanwinkleiglTest": "DATA_DIR/v3/v3convert_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_test_DXSMVALID_no-naive.csv.gz",
    "v1vanwinkleiglTestCC": "DATA_DIR/v3/v3convert_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_test_DXSMVALID_ConsCys_no-naive.csv.gz",
    "v1vanwinkleiglTrain": "DATA_DIR/v3/v3convert_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_no-naive.csv.gz",
    "v1vanwinkleiglTrainCC": "DATA_DIR/v3/v3convert_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive.csv.gz",
    "v1vanwinkleiglTrainCC50k": "DATA_DIR/v3/v3convert_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_50k.csv.gz",
    "v1vanwinkleiglTrainCC100k": "DATA_DIR/v3/v3convert_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_100k.csv.gz",
    "v1vanwinkleiglTrainCC250k": "DATA_DIR/v3/v3convert_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_250k.csv.gz",
    "v1vanwinkleiglTrainCC500k": "DATA_DIR/v3/v3convert_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_500k.csv.gz",
    "v1vanwinklelightTrainCCWithNaive100k": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_downsample_50k_CONCAT_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_downsample_50k.csv.gz",
    "v1vanwinklelightTrainCC100k": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_50k_CONCAT_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_50k.csv.gz",
    "v1vanwinklelightTrainCC200k": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_100k_CONCAT_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_100k.csv.gz",
    "v1vanwinklelightTrainCC500k": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_250k_CONCAT_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_250k.csv.gz",
    "v1vanwinklelightTrainCC1m": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_500k_CONCAT_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_500k.csv.gz",
    "v1vanwinklelightTrain1m": "DATA_DIR/v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_no-naive_downsample_500k_CONCAT_v3convert_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_no-naive_downsample_500k.csv.gz",
    "v1vanwinkleheavyTest": "DATA_DIR/v3/v3convert_vanwinkle-170-igh_pcp_2025-03-05_MASKED_NI_test_no-naive_DXSMVALID.csv.gz",
    "v1vanwinkleheavyTestCC": "DATA_DIR/v3/v3convert_vanwinkle-170-igh_pcp_2025-03-05_MASKED_NI_test_no-naive_DXSMVALID_ConsCys.csv.gz",
    "v1vanwinkleheavyTrain": "DATA_DIR/v3/v3convert_vanwinkle-170-igh_pcp_2025-03-05_MASKED_NI_train_no-naive_DXSMVALID.csv.gz",
    "v1vanwinkleheavyTrainCC": "DATA_DIR/v3/v3convert_vanwinkle-170-igh_pcp_2025-03-05_MASKED_NI_train_no-naive_DXSMVALID_ConsCys.csv.gz",
    "v2kimTrain": "DATA_DIR/v3/kim-zhou-scv2-vacc_igh_pcp_2025-05-19_MASKED_NI_train_no-naive_DXSMVALID.csv.gz",
    "v2kimTest": "DATA_DIR/v3/kim-zhou-scv2-vacc_igh_pcp_2025-05-19_MASKED_NI_test_no-naive_DXSMVALID.csv.gz",
    # Simulated datasets from here down
    "jaffeTangDnsmSim50k": "DATA_DIR/simulations/v3/v3convert_dnsm_jaffe+tang_SIM_v2tang-2025-6-3_NI_no-naive_CONCAT_dnsm_jaffe+tang_SIM_v1jaffebulk-2025-6-3_NI_no-naive_downsample_50k.csv.gz",
    "jaffeTangDnsmSim100k": "DATA_DIR/simulations/v3/v3convert_dnsm_jaffe+tang_SIM_v2tang-2025-6-3_NI_no-naive_CONCAT_dnsm_jaffe+tang_SIM_v1jaffebulk-2025-6-3_NI_no-naive_downsample_100k.csv.gz",
    "jaffeTangDnsmSim250k": "DATA_DIR/simulations/v3/v3convert_dnsm_jaffe+tang_SIM_v2tang-2025-6-3_NI_no-naive_CONCAT_dnsm_jaffe+tang_SIM_v1jaffebulk-2025-6-3_NI_no-naive_downsample_250k.csv.gz",
    "jaffeTangDnsmSim": "DATA_DIR/simulations/v3/v3convert_dnsm_jaffe+tang_SIM_v2tang-2025-6-3_NI_no-naive_CONCAT_dnsm_jaffe+tang_SIM_v1jaffebulk-2025-6-3_NI_no-naive.csv.gz",
    "rodriguezDnsmSim": "DATA_DIR/simulations/v3/v3convert_dnsm_jaffe+tang_SIM_rodriguez-6-2-25_NI_no-naive.csv.gz",
    "heavyDasmPairedSim402k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v2tang-2025-6-17_NI_no-naive_ConsCys_downsample_337k_CONCAT_dasm_paired_SIM_v1vanwinkleheavy-2025-6-17_NI_no-naive_ConsCys_train_downsample_65k.csv.gz",
    "heavyDasmPairedSim201k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v2tang-2025-6-17_NI_no-naive_ConsCys_downsample_337k_CONCAT_dasm_paired_SIM_v1vanwinkleheavy-2025-6-17_NI_no-naive_ConsCys_train_downsample_65k_downsample_201k.csv.gz",
    "heavyDasmPairedSim100k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v2tang-2025-6-17_NI_no-naive_ConsCys_downsample_337k_CONCAT_dasm_paired_SIM_v1vanwinkleheavy-2025-6-17_NI_no-naive_ConsCys_train_downsample_65k_downsample_100k.csv.gz",
    "heavyDasmPairedSim40k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v2tang-2025-6-17_NI_no-naive_ConsCys_downsample_337k_CONCAT_dasm_paired_SIM_v1vanwinkleheavy-2025-6-17_NI_no-naive_ConsCys_train_downsample_65k_downsample_40k.csv.gz",
    "pairedDasmPairedSim108k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v2jaffePaired-2025-6-17_NI_no-naive_ConsCys_HL_downsample_108k.csv.gz",
    "pairedDasmPairedSim54k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v2jaffePaired-2025-6-17_NI_no-naive_ConsCys_HL_downsample_108k_downsample_54k.csv.gz",
    "pairedDasmPairedSim27k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v2jaffePaired-2025-6-17_NI_no-naive_ConsCys_HL_downsample_108k_downsample_27k.csv.gz",
    "pairedDasmPairedSim11k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v2jaffePaired-2025-6-17_NI_no-naive_ConsCys_HL_downsample_108k_downsample_11k.csv.gz",
    "lightDasmPairedSim518k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v1vanwinkleigk-2025-6-17_NI_no-naive_ConsCys_train_downsample_259k_CONCAT_dasm_paired_SIM_v1vanwinkleigl-2025-6-17_NI_no-naive_ConsCys_train.csv.gz",
    "lightDasmPairedSim260k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v1vanwinkleigk-2025-6-17_NI_no-naive_ConsCys_train_downsample_259k_CONCAT_dasm_paired_SIM_v1vanwinkleigl-2025-6-17_NI_no-naive_ConsCys_train_downsample_260k.csv.gz",
    "lightDasmPairedSim130k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v1vanwinkleigk-2025-6-17_NI_no-naive_ConsCys_train_downsample_259k_CONCAT_dasm_paired_SIM_v1vanwinkleigl-2025-6-17_NI_no-naive_ConsCys_train_downsample_130k.csv.gz",
    "lightDasmPairedSim52k": "DATA_DIR/simulations/v3/dasm_paired_SIM_v1vanwinkleigk-2025-6-17_NI_no-naive_ConsCys_train_downsample_259k_CONCAT_dasm_paired_SIM_v1vanwinkleigl-2025-6-17_NI_no-naive_ConsCys_train_downsample_52k.csv.gz",
    "rodriguezDasmPairedSim": "DATA_DIR/simulations/v3/dasm_paired_SIM_v1rodriguez-2025-6-17_NI_no-naive_ConsCys.csv.gz",
    "vanwinkleigkTestDasmPairedSim": "DATA_DIR/simulations/v3/dasm_paired_SIM_v1vanwinkleigk-2025-6-17_NI_no-naive_ConsCys_test.csv.gz",
    "vanwinkleiglTestDasmPairedSim": "DATA_DIR/simulations/v3/dasm_paired_SIM_v1vanwinkleigl-2025-6-17_NI_no-naive_ConsCys_test.csv.gz",
}


_vanwinkle_light_holdouts = [
    "RG1895-igk",
    "110044845-igk",
    "CE0007814-igk",
    "RG3355-igk",
    "110040958-igk",
    "SCT6338383-igk",
    "888508776-igk",
    "888634432-igk",
    "CE0009275-igk",
    "CE0006205-igk",
    "SCT8789722-igk",
    "CE0005662-igk",
    "CE0006233-igk",
    "888976935-igk",
    "CE0006943-igk",
    "888956204-igk",
    "CE0005802-igk",
    "CE0005988-igk",
    "RG3578-igk",
    "CE0007908-igk",
    "888179544-igk",
    "CE0007408-igk",
    "110044355-igk",
    "CE0007424-igk",
    "110046617-igk",
    "888921048-igk",
    "CE0008202-igk",
    "RG3105-igk",
    "RG2105-igk",
    "RG2155-igk",
    "110042592-igk",
    "888262983-igk",
    "888624327-igk",
    "22301641N-igl",
    "RG3578-igl",
    "CE0006524-igl",
    "SCT8995047-igl",
    "110040887-igl",
    "110042606-igl",
    "888777004-igl",
    "SCT8789722-igl",
    "CE0008817-igl",
    "CE0005876-igl",
    "889018176-igl",
    "888634432-igl",
    "RG2195-igl",
    "110040277-igl",
    "888816126-igl",
    "888402542-igl",
    "110044782-igl",
    "RG2251-igl",
    "RG1265-igl",
    "RG1083-igl",
    "CE0007814-igl",
    "CE0008593-igl",
    "CE0008682-igl",
    "RG2685-igl",
    "110040851-igl",
    "110046617-igl",
    "CE0006481-igl",
    "RG1791-igl",
    "RG2037-igl",
    "RG3844-igl",
    "CE0006547-igl",
    "CE0005604-igl",
    "SCT2794524-igl",
]
_vanwinkle_heavy_holdouts = [
    "110040277-igh",
    "888408987-igh",
    "RG2617-igh",
    "CE0008114-igh",
    "888402542-igh",
    "110042606-igh",
    "CE0006829-igh",
    "CE0008264-igh",
    "RG3987-igh",
    "RG3945-igh",
    "110044118-igh",
    "RG2459-igh",
    "RG1367-igh",
    "RG1895-igh",
    "CE0007129-igh",
    "SCT4168329-igh",
    "888179544-igh",
    "RG1578-igh",
    "CE0007865-igh",
    "CE0006705-igh",
    "RG2195-igh",
    "888927334-igh",
    "CE0009275-igh",
    "110044376-igh",
    "888703260-igh",
    "CE0005797-igh",
    "888174491-igh",
]

holdout_dict = {
    "v1tang": ["B11", "B20", "B21", "CLL1729"],
    "v1tangSingletons": ["B11", "B20", "B21", "CLL1729"],
    "v1tangWithNaive": ["B11", "B20", "B21", "CLL1729"],
    "v1jaffe": ["d2"],
    "v1jaffePaired": ["d2"],
    "v2jaffePaired": ["d2"],
    "v1vanwinkleigkTrain": _vanwinkle_light_holdouts,
    "v1vanwinkleiglTrain": _vanwinkle_light_holdouts,
    "v1vanwinklelightTrain": _vanwinkle_light_holdouts,
    "v1vanwinkleheavyTrain": _vanwinkle_heavy_holdouts,
    "v1rodriguezPrimers": [
        "sample-igg-B-39",
        "sample-igg-B-65",
        "sample-igg-W-114",
        "sample-igg-W-62",
        "sample-igg-B-55",
        "sample-igg-W-110",
        "sample-igg-B-4",
        "sample-igg-B-36",
        "sample-igg-W-38",
        "sample-igg-B-33",
        "sample-igg-W-67",
        "sample-igg-W-60",
        "sample-igg-B-105",
        "sample-igg-B-75",
        "sample-igg-W-51",
        "sample-igg-B-74",
        "sample-igg-W-90",
        "sample-igg-W-115",
        "sample-igg-W-99",
    ],
    "v2kimTrain": ["368-02a"],
    "jaffeTangDnsmSim": ["d2"] + ["B11", "B20", "B21", "CLL1729"],
    "heavyDasmPairedSim": ["B11", "B20", "B21", "CLL1729"] + _vanwinkle_heavy_holdouts,
    "pairedDasmPairedSim": ["d2"],
    "lightDasmPairedSim": _vanwinkle_light_holdouts,
}
suffixes = (
    "50k",
    "100k",
    "200k",
    "250k",
    "500k",
    "1m",
    "402k",
    "201k",
    "100k",
    "40k",
    "108k",
    "54k",
    "27k",
    "11k",
    "518k",
    "260k",
    "130k",
    "52k",
    "CC",
    "WithN",
    "CC50k",
    "CC100k",
    "CC200k",
    "CC250k",
    "CC500k",
    "CC1m",
)
holdout_dict.update(
    {k + suffix: v for suffix in suffixes for k, v in holdout_dict.items()}
)


anarci_dict = {
    "v1rodriguez": "DATA_DIR/v3/anarci/rodriguez-airr-seq-race-prod-NoWinCheck_igh_imgt.csv",
    "v1tang": "DATA_DIR/v3/anarci/tang-deepshm-prod-NoWinCheck_igh_imgt.csv",
    "v1tangSingletons": "DATA_DIR/v3/anarci/tang-deepshm-prod-NoWinCheck_igh_imgt.csv",
    "v1tang50k": "DATA_DIR/v3/anarci/tang-deepshm-prod-NoWinCheck_igh_imgt.csv",
    "v1jaffe": "DATA_DIR/v3/anarci/wyatt-10x-1p5m_fs-all-NoWinCheck_igh_imgt.csv",
    "v1flairr": "DATA_DIR/v3/anarci/ford-flairr-seq-prod-NoWinCheck-UnmutInv_igh_imgt.csv",
    "v1jaffePaired": {
        "heavy": "DATA_DIR/v3/anarci/wyatt-10x-1p5m_fs-all-NoWinCheck-UnmutInv-GTR-paired-igh_imgt.csv",
        "light": "DATA_DIR/v3/anarci/wyatt-10x-1p5m_fs-all-NoWinCheck-UnmutInv-GTR-paired-igk-igl_imgt.csv",
    },
    "v1vanwinkleigkTest": {
        "light": "DATA_DIR/v3/anarci/vanwinkle-170-NoWinCheck_igk_imgt.csv",
    },
    "v1vanwinkleiglTest": {
        "light": "DATA_DIR/v3/anarci/vanwinkle-170-NoWinCheck_igl_imgt.csv",
    },
    "v1vanwinkleheavyTest": "DATA_DIR/v3/anarci/vanwinkle-170-NoWinCheck_igh_imgt.csv",
    "v2kimTest": "DATA_DIR/anarci/kim-zhou-scv2-vacc_igh_imgt.csv",
    "rodriguezDnsmSim": "DATA_DIR/v3/anarci/rodriguez-airr-seq-race-prod-NoWinCheck_igh_imgt.csv",
    "rodriguezDasmPairedSim": "DATA_DIR/v3/anarci/rodriguez-airr-seq-race-prod-NoWinCheck_igh_imgt.csv",
    "vanwinkleigkTestDasmPairedSim": {
        "light": "DATA_DIR/v3/anarci/vanwinkle-170-NoWinCheck_igk_imgt.csv",
    },
    "vanwinkleiglTestDasmPairedSim": {
        "light": "DATA_DIR/v3/anarci/vanwinkle-170-NoWinCheck_igl_imgt.csv",
    },
}

anarci_dict.update(
    {k + suffix: v for suffix in suffixes for k, v in anarci_dict.items()}
)

# Test datasets do not have the same naming convention -- the different
# versions do not come from the same dataset source.
anarci_dict.update(
    {
        "tstWithN": "DATA_DIR/v3/anarci/tang-deepshm-prod-NoWinCheck_igh_imgt.csv",
        "tst": "DATA_DIR/anarci/tst_imgt.csv",
        "tstPaired": {
            "heavy": "DATA_DIR/v3/anarci/wyatt-10x-1p5m_fs-all-NoWinCheck-UnmutInv-GTR-paired-igh_imgt.csv",
            "light": "DATA_DIR/v3/anarci/wyatt-10x-1p5m_fs-all-NoWinCheck-UnmutInv-GTR-paired-igk-igl_imgt.csv",
        },
    }
)

# Assume unlabeled values are for heavy chains.
anarci_dict = {
    name: ({"heavy": val} if isinstance(val, str) else val)
    for name, val in anarci_dict.items()
}


def _nested_dict_value_map(func, d):
    return {
        k: _nested_dict_value_map(func, v) if isinstance(v, dict) else func(v)
        for k, v in d.items()
    }


dataset_dict = {name: localify(path) for name, path in dataset_dict.items()}
anarci_dict = _nested_dict_value_map(localify, anarci_dict)

naughty_clonal_families = {
    "v1flairr": [("sample-igg-1013", "585")],
    "v1rodriguez": [("sample-igg-SC-18", "440")],
    "v1rodriguezCC": [("sample-igg-SC-18", "440")],
    "v1tang": [("B13", "77"), ("CLL1697", "43875"), ("CLL2056", "18535")],
    "v1tangCC": [("B13", "77"), ("CLL1697", "43875"), ("CLL2056", "18535")],
    "v1tangWithN": [("B13", "77"), ("CLL1697", "43875"), ("CLL2056", "18535")],
    "tstWithN": [("B13", "77"), ("CLL1697", "43875"), ("CLL2056", "18535")],
    "v1jaffe": [("d4", "213393")],
    "v1jaffeCC": [("d4", "213393")],
}


class _ValidPCPFilterFunc:

    def __init__(self, parent_func, child_func):
        self.parent_func = parent_func
        self.child_func = child_func

    def __call__(self, row):
        try:
            # Although this is not exactly how heavy and light sequences are combined
            # for model application, it should work for checking which ones are valid.
            assert_pcp_valid(
                self.parent_func(row),
                self.child_func(row),
            )
            return True
        except ValueError:
            return False


def _default_get_row_parent(row):
    return (
        row["parent_heavy"][: (len(row["parent_heavy"]) // 3) * 3] + row["parent_light"]
    )


def _default_get_row_child(row):
    return (
        row["child_heavy"][: (len(row["parent_heavy"]) // 3) * 3] + row["child_light"]
    )


def filter_valid_pcps(
    pcp_df,
    get_row_parent=_default_get_row_parent,
    get_row_child=_default_get_row_child,
    parallelize=True,
    force_parallel=None,
):
    """Filter out PCPs whose ambiguities make them unusable for inference.

    Args:
        pcp_df: A DataFrame containing the pcp data.
        get_row_parent: A function that takes a row and returns the parent sequence.
        get_row_child: A function that takes a row and returns the child sequence.
        parallelize: If True, use parallel processing to filter the DataFrame. If there are too
            few rows or too few processors, the call will not be parallelized regardless of this value.
        force_parallel: If an integer is provided, it will force the use of that many parallel processes, regardless of the value of parallelize.


    In order to run this function in parallel, the functions passed to
    get_row_parent and get_row_child must be pickleable.

    Modifies the passed pcp_df in-place and returns it.
    """
    print(
        "Filtering PCPs without mutations, or whose ambiguities make them unusable for inference"
    )

    filter_func = _ValidPCPFilterFunc(get_row_parent, get_row_child)

    if parallelize or force_parallel is not None:
        pcp_df = pcp_df[
            parallel_df_apply(
                pcp_df,
                filter_func,
                force_parallel=force_parallel,
                use_progress_apply=True,
            )
        ]
    else:
        pcp_df = pcp_df[pcp_df.progress_apply(filter_func, axis=1)]
    return pcp_df


def pcp_df_of_nickname(
    dataset_name,
    sample_count=None,
    add_shm_outputs=False,
    device=None,
    filter_invalid_pcps=False,
):
    """Load a pcp_df from the dataset_dict, optionally adding SHM model outputs.

    sample_count: If not None, downsample to this number.
    add_shm_outputs: If True, add SHM model outputs.
    device: If not None, use this device for the SHM model.
    """
    print(f"Loading {dataset_dict[dataset_name]}")

    pcp_df = load_pcp_df(
        dataset_dict[dataset_name],
        sample_count=sample_count,
    )

    naughty_key = next(
        (key for key in naughty_clonal_families if dataset_name.startswith(key)), None
    )

    if naughty_key:
        naughty_families = naughty_clonal_families[naughty_key]
        print(
            f"Filtering out problematic pairs of sample_id and clonal families {naughty_families}"
        )

        for sample_id, family in naughty_families:
            pcp_df = pcp_df[
                ~((pcp_df["sample_id"] == sample_id) & (pcp_df["family"] == family))
            ]

    if filter_invalid_pcps:
        pcp_df = filter_valid_pcps(pcp_df)

    if add_shm_outputs:
        neutral_model_name = DEFAULT_NEUTRAL_MODEL
        warn(
            f"Using {neutral_model_name}. Make sure this is an appropriate choice for your model, "
            "or use `netam.framework.add_shm_model_outputs_to_pcp_df` directly to provide the "
            "correct neutral model."
        )
        neutral_crepe = pretrained.load(neutral_model_name, device="cpu")
        pcp_df = add_shm_model_outputs_to_pcp_df(pcp_df, neutral_crepe)
    pcp_df.reset_index(drop=True, inplace=True)
    return pcp_df


def train_val_df_of_nickname(
    dataset_name, sample_count=None, device=None, add_shm_outputs=False
):
    """Load a pcp_df from the dataset_dict, add an "in_train" column marking training
    samples, and return the resulting dataframe.

    If the dataset_name is in holdout_dict, mark the holdout samples as not in training.
    Otherwise, use the first 80% of the samples for training.
    """
    pcp_df = pcp_df_of_nickname(
        dataset_name,
        sample_count=sample_count,
        device=device,
        add_shm_outputs=add_shm_outputs,
    )
    pcp_df["in_train"] = True
    if dataset_name in holdout_dict:
        holdout_samples = holdout_dict[dataset_name]
        pcp_df.loc[pcp_df["sample_id"].isin(holdout_samples), "in_train"] = False
        holdout_row_count = pcp_df["sample_id"].isin(holdout_samples).sum()
        print(f"Holdout samples for {dataset_name}: {holdout_samples}")
        print(f"This represents {holdout_row_count/len(pcp_df):.2%} of the data.")
    else:
        train_frac = 0.8
        print(
            f"No holdout samples for {dataset_name}. Using the first {train_frac:.2%} for training."
        )
        train_len = int(train_frac * len(pcp_df))
        pcp_df.loc[train_len:, "in_train"] = False
    return pcp_df


def parse_mixing_instructions(multiname):
    """Parse a multiname string to extract fine-tuning mixing instructions.

    Parameters:
    -----------
    multiname : str
        The multiname string which might contain mixing instructions

    Returns:
    --------
    tuple
        (base_multiname, mixing_spec) where:
        - base_multiname is the part before the first @ symbol
        - mixing_spec is None if no mixing is specified, or a tuple (ratio, ft_multiname)
    """
    if "@" in multiname:
        parts = multiname.split("@")
        if len(parts) != 3:
            raise ValueError(
                f"Fine-tuning format should be 'base_datasets@ratio@ft_datasets', got {multiname}"
            )
        base_multiname, ratio_str, ft_multiname = parts
        return base_multiname, (ratio_str, ft_multiname)
    else:
        return multiname, None


def combine_dfs(multiname, df_loader, mixing_spec=None):
    """Combine multiple dataframes based on the multiname specification.

    Parameters:
    -----------
    multiname : str
        String specification of datasets to combine, using "+" notation
    df_loader : callable
        Function that takes a single name and returns a dataframe.
        Example: lambda name: train_val_df_of_nickname(name, sample_count=1000)
    mixing_spec : tuple, optional
        If provided, specifies fine-tuning mixing parameters (ratio, ft_multiname)

    Returns:
    --------
    pandas.DataFrame
        Combined dataframe
    """
    names = multiname.split("+")
    dfs = []

    for name in names:
        # Use the provided loader function to load each dataframe.
        df = df_loader(name)
        if df is not None and len(df) > 0:
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No valid dataframes found for multiname: {multiname}")

    combined_df = pd.concat(dfs)

    # If mixing_spec is provided, mix with fine-tuning data.
    if mixing_spec is not None:
        ratio, ft_multiname = mixing_spec
        # Parse ratio (e.g., "1/3" -> 1/3).
        if isinstance(ratio, str) and "/" in ratio:
            num, denom = ratio.split("/")
            ratio = float(num) / float(denom)
        else:
            ratio = float(ratio)

        # Load fine-tuning dataframes using the same loader function.
        ft_df = combine_dfs(ft_multiname, df_loader)

        # Mix dataframes according to the ratio.
        combined_df = mix_dfs(combined_df, ft_df, ratio)

    return combined_df


def mix_dfs(base_df, ft_df, ratio):
    """Mix base dataframe with fine-tuning dataframe according to specified ratio.

    The function interleaves rows from base_df and ft_df. When ft_df runs out,
    it cycles back to the beginning. Stops when base_df is exhausted.

    Parameters:
    -----------
    base_df : pandas.DataFrame
        Base dataframe
    ft_df : pandas.DataFrame
        Fine-tuning dataframe
    ratio : float
        Proportion of fine-tuning data to mix in

    Returns:
    --------
    pandas.DataFrame
        Mixed dataframe
    """
    if len(ft_df) == 0:
        return base_df

    if len(base_df) == 0:
        return ft_df

    result_rows = []
    ft_index = 0

    # Process all base samples to ensure base_df is exhausted.
    for _, base_row in base_df.iterrows():
        # Add the base sample.
        result_rows.append(base_row)

        # Add fine-tuning samples.
        ft_to_add = int(ratio)
        fractional_part = ratio - ft_to_add

        # Handle probabilistic addition for fractional part.
        if fractional_part > 0 and random.random() < fractional_part:
            ft_to_add += 1

        # Add the determined number of fine-tuning samples.
        for _ in range(ft_to_add):
            # Loop around if we reach the end of ft_df.
            if ft_index >= len(ft_df):
                ft_index = 0

            result_rows.append(ft_df.iloc[ft_index])
            ft_index += 1

    return pd.DataFrame(result_rows)


def train_val_df_of_multiname(
    multiname, sample_count=None, device=None, add_shm_outputs=False
):
    """Load dataframes specified by multiname, add an "in_train" column marking training
    samples.

    The multiname can be in two formats:
    1. Simple concatenation: "dataset1+dataset2+dataset3"
    2. Fine-tuning format: "base_dataset1+base_dataset2@ratio@ft_dataset1+ft_dataset2"
        Note that here + takes precedence over @, so the base datasets are combined before mixing.

    Parameters:
    -----------
    multiname : str
        The multiname string specifying dataframes to load
    sample_count : int, optional
        If specified, downsample to this number
    device : str, optional
        Device to use for SHM model
    add_shm_outputs : bool, optional
        Whether to add SHM model outputs

    Returns:
    --------
    pandas.DataFrame
        Combined dataframe with train/val split
    """
    # Validate dataset name for filesystem safety
    validate_dataset_name_for_filesystem_safety(multiname)

    # Parse the multiname to check for fine-tuning format.
    base_multiname, mixing_spec = parse_mixing_instructions(multiname)

    # Define the loader function to pass to combine_dfs.
    loader_func = lambda name: train_val_df_of_nickname(
        name, sample_count=sample_count, device=device, add_shm_outputs=add_shm_outputs
    )

    # Use the general combining function.
    return combine_dfs(base_multiname, loader_func, mixing_spec)


def train_val_datasets_of_multiname(
    dataset_cls,
    multiname,
    model_known_token_count,
    neutral_model_name,
    sample_count=None,
    device=None,
    multihit_model_name=None,
):
    """Splits multiname by "+", splits them into train and validation datasets, and
    gives back the resulting datasets."""
    pcp_df = train_val_df_of_multiname(
        multiname, sample_count=sample_count, device=device
    )
    neutral_crepe = pretrained.load(neutral_model_name, device="cpu")
    pcp_df = add_shm_model_outputs_to_pcp_df(
        pcp_df,
        neutral_crepe,
    )
    train_dataset, val_dataset = dataset_cls.train_val_datasets_of_pcp_df(
        pcp_df,
        model_known_token_count,
        branch_length_multiplier=5.0,
        multihit_model=pretrained.load_multihit(multihit_model_name, device=device),
    )
    print(
        f"we have {len(train_dataset)} training examples and {len(val_dataset)} validation examples"
    )
    return pcp_df, train_dataset, val_dataset


def pcp_df_of_multiname(
    multiname,
    sample_count=None,
    device=None,
    add_shm_outputs=False,
):
    """Load and concat dataframes specified by multiname.

    The multiname can be in two formats:
    1. Simple concatenation: "dataset1+dataset2+dataset3"
    2. Fine-tuning format: "base_dataset1+base_dataset2@ratio@ft_dataset1+ft_dataset2"

    Parameters:
    -----------
    multiname : str
        The multiname string specifying dataframes to load
    sample_count : int, optional
        If specified, downsample to this number
    device : str, optional
        Device to use for SHM model
    add_shm_outputs : bool, optional
        Whether to add SHM model outputs

    Returns:
    --------
    pandas.DataFrame
        Combined dataframe
    """
    # Parse the multiname to check for fine-tuning format.
    base_multiname, mixing_spec = parse_mixing_instructions(multiname)

    # Define the loader function to pass to combine_dfs.
    loader_func = lambda name: pcp_df_of_nickname(
        name, sample_count=sample_count, device=device, add_shm_outputs=add_shm_outputs
    )

    # Use the general combining function.
    return combine_dfs(base_multiname, loader_func, mixing_spec)


def dataset_of_pcp_df(
    dataset_cls,
    pcp_df,
    model_known_token_count,
    neutral_model_name,
    device=None,
    multihit_model_name=None,
    add_shm_outputs=True,
):
    if add_shm_outputs:
        neutral_crepe = pretrained.load(neutral_model_name, device="cpu")
        pcp_df = add_shm_model_outputs_to_pcp_df(
            pcp_df,
            neutral_crepe,
        )

    pcp_df["in_train"] = False
    _, dataset = dataset_cls.train_val_datasets_of_pcp_df(
        pcp_df,
        model_known_token_count,
        branch_length_multiplier=5.0,
        multihit_model=pretrained.load_multihit(multihit_model_name, device=device),
    )
    return dataset


def dataset_of_multiname(
    dataset_cls,
    multiname,
    model_known_token_count,
    neutral_model_name,
    sample_count=None,
    device=None,
    multihit_model_name=None,
):
    pcp_df = pcp_df_of_multiname(multiname, sample_count=sample_count, device=device)
    dataset = dataset_of_pcp_df(
        dataset_cls,
        pcp_df,
        model_known_token_count,
        neutral_model_name,
        device=device,
        multihit_model_name=multihit_model_name,
    )
    return pcp_df, dataset


def _check_train_holdouts():
    for nickname, dataset_path in dataset_dict.items():
        if nickname not in holdout_dict:
            print(f"No holdout samples for {nickname}.")
            continue
        pcp_df = load_pcp_df(dataset_path)
        holdout_samples = holdout_dict[nickname]
        holdout_row_count = pcp_df["sample_id"].isin(holdout_samples).sum()
        print(
            f"Holdout samples for {nickname}: represent {holdout_row_count/len(pcp_df):.2%} of the data."
        )
