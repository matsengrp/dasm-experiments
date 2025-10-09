import os
import random
import argparse
import shutil
import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
import math
from IPython.display import display

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.DSSP import DSSP
from Bio import PDB
from Bio.SeqUtils import seq1

from netam.framework import load_crepe


def display_all(*args):
    """Display multiple objects with full visibility of Pandas DataFrames."""
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        for arg in args:
            display(arg)


def df_remove_index_cols(df):
    df = df.drop(
        columns=[col for col in df.columns if "Unnamed" in col or col == "index"],
        errors="ignore",
    )
    return df


def df_get_columns(df, match_term):
    cols = [x for x in df.columns if (x.find(match_term) >= 0)]
    return cols


def is_empty(val):
    return val is None or val == "" or val == {} or val == [] or pd.isna(val)


def dict_remove_empty(dict_old):
    dict_new = {k: v for k, v in dict_old.items() if not is_empty(v)}
    return dict_new


class ColorPrinter:
    class colors:
        BLACK = "\033[30m"
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"

        BG_BLACK = "\033[40m"
        BG_RED = "\033[41m"
        BG_GREEN = "\033[42m"
        BG_YELLOW = "\033[43m"
        BG_BLUE = "\033[44m"
        BG_MAGENTA = "\033[45m"
        BG_CYAN = "\033[46m"
        BG_WHITE = "\033[47m"

        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"
        RESET = "\033[0m"

    @staticmethod
    def bash_color_code(R=255, G=255, B=255):
        code = f"\033[38;2;{R};{G};{B};0m"
        return code

    @staticmethod
    def print(*args, color=None, bg_color=None, style=None, sep=" ", end="\n"):
        color_code = ""
        if color:
            color_code += color
        if bg_color:
            color_code += bg_color
        if style:
            color_code += style
        color_reset = ""
        if color_code != "":
            color_reset = ColorPrinter.colors.RESET
        message = sep.join(map(str, args))
        print(f"{color_code}{message}{color_reset}", end=end)

    @staticmethod
    def set_color(color):
        print(f"{color}")

    @staticmethod
    def unset_color():
        print(f"{ColorPrinter.colors.RESET}")


colors = ColorPrinter.colors
cprint = ColorPrinter.print
cprint_set_color = ColorPrinter.set_color
cprint_unset_color = ColorPrinter.unset_color


class DmsViz_Utility:
    # options
    # always download, regardless if local file already exists.
    REDOWNLOAD_ALL = False
    # which online database to retrieve pdbs from. options: (rcsb | opig)
    DOWNLOAD_PDB_FROM = "opig"
    USE_DSSP_FOR_SEQ = True
    CHAIN_TYPES = ["H", "L"]
    NUM_SCHEME = "rcsb"
    SITECOUNT_NUM_SCHEME = "opig-imgt"
    ALL_NUM_SCHEMES = ["opig-imgt", "opig-chothia", "rcsb"]
    MIN_NUM_SCHEME = 1
    MAX_NUM_SCHEME = 128
    # method for calculating relative ASA (new method removes unused chainids BEFORE using mkdssp)
    CHAINID_FILTER_METHOD = "new"
    ALWAYS_REBUILD_CSV = False
    UNIQUE_PDBIDS = False
    PDBS_RANGE = None  # Build table over pdb id range. `None` covers all pdbs.
    # pdb parser settings
    PDB_PARSER_PERMISSIVE = True
    PDB_PARSER_QUIET = False
    # antibody regions
    ALL_REGIONS = {
        "CDR1": (27, 38),
        "CDR2": (56, 65),
        "CDR3": (105, 117),
        "FR1": (0, 26),
        "FR2": (39, 55),
        "FR3": (66, 104),
        "FR4": (118, 128),
    }
    CDR_REGIONS = {k: v for k, v in ALL_REGIONS.items() if k.startswith("CDR")}
    FR_REGIONS = {k: v for k, v in ALL_REGIONS.items() if k.startswith("FR")}

    # default input paths
    SHARED_DIR = "/fh/fast/matsen_e/shared/bcr-mut-sel"
    DNSM_MODELS_DIR = f"{SHARED_DIR}/dnsm/dnsm-experiments-1/dnsm-train/trained_models"
    SABDAB_DIR = f"{SHARED_DIR}/sabdab"
    PDB_DB_DIR = f"{SABDAB_DIR}/pdb-db"
    SABDAB_PATH = f"{SABDAB_DIR}/sabdab_summary_for_dnsm.tsv"
    DNSM_MODEL_NAME = "dnsm_1m-v1jaffe+v1tang-joint"
    # default output paths
    OUTPUT_DIR = f"{SABDAB_DIR}/_output"
    TEMP_DIR = f"{SABDAB_DIR}/_temp"
    LOG_PATH = f"{TEMP_DIR}/build_pdb_table.log"

    def __init__(
        self,
        output_dir=OUTPUT_DIR,
        temp_dir=TEMP_DIR,
        sabdab_path=SABDAB_PATH,
        pdb_db_dir=PDB_DB_DIR,
        dnsm_models_dir=DNSM_MODELS_DIR,
        dnsm_model_name=DNSM_MODEL_NAME,
    ):
        self.OUTPUT_DIR = output_dir
        self.TEMP_DIR = temp_dir
        self.SABDAB_PATH = sabdab_path
        self.PDB_DB_DIR = pdb_db_dir
        self.DNSM_MODELS_DIR = dnsm_models_dir
        self.DNSM_MODEL_NAME = dnsm_model_name
        self.DNSM_MODEL_PATH = f"{self.DNSM_MODELS_DIR}/{self.DNSM_MODEL_NAME}"
        self.PDB_CSV_PATH = self.update_pdbs_csv_path()

        self.sabdab_df = None
        self.pdbs_df = None

    # paths

    def update_dnsm_model(self, dnsm_model_name):
        self.DNSM_MODEL_NAME = dnsm_model_name
        self.DNSM_MODEL_PATH = f"{self.DNSM_MODELS_DIR}/{self.DNSM_MODEL_NAME}"
        self.PDB_CSV_PATH = self.update_pdbs_csv_path()

    def update_pdbs_csv_path(self):
        self.PDB_CSV_NAME = f"pdb-db.{self.DNSM_MODEL_NAME}.ALL.csv"
        if self.PDBS_RANGE:
            self.PDB_CSV_NAME = self.PDB_CSV_NAME.replace(
                ".ALL.csv", f".{self.PDBS_RANGE[0]}-{self.PDBS_RANGE[1]}.csv"
            )
        self.PDB_CSV_PATH = f"{self.OUTPUT_DIR}/{self.PDB_CSV_NAME}"
        return self.PDB_CSV_PATH

    def find_pdbs_csv_path(self):
        is_found = os.path.exists(f"{self.PDB_CSV_PATH}")
        if is_found:
            self.PDB_CSV_PATH = f"{self.PDB_CSV_PATH}"
        return is_found

    # file loading

    def load_sabdab_table(self):
        self.sabdab_df = pd.read_table(self.SABDAB_PATH)
        self.sabdab_df = self.sabdab_df[
            ["organism", "pdbid", "abid", "vb", "jb", "chainseq_b"]
        ]
        return self.sabdab_df

    def load_model(self):
        self.dnsm_model = load_crepe(self.DNSM_MODEL_PATH).model
        return self.dnsm_model

    def save_pdbs_table(self):
        pdbs_csv_path_found = self.find_pdbs_csv_path()
        if not pdbs_csv_path_found or self.ALWAYS_REBUILD_CSV:
            self.DNSM_MODEL_PATH = f"{self.DNSM_MODELS_DIR}/{self.DNSM_MODEL_NAME}"
            self.pdbs_df = self.build_pdbs_table()
            self.pdbs_df.to_csv(f"{self.PDB_CSV_PATH}", index=False)
            print(f"pdbs_df saved to: {self.PDB_CSV_PATH}")
        else:
            print(f"pdbs_df already found at: {self.PDB_CSV_PATH}")
            self.load_pdbs_table()
        return self.pdbs_df

    def load_pdbs_table(self):
        pdbs_csv_path_found = self.find_pdbs_csv_path()
        if not pdbs_csv_path_found:
            print(f"[ERROR] pdbs_csv path not found: {self.PDB_CSV_PATH}")
            return None
        self.pdbs_df = pd.read_csv(self.PDB_CSV_PATH)
        return self.pdbs_df

    # dms table builder

    def build_pdbs_table(self, log_path="_output/build_pdb_table.log"):
        self.pdbs_df = pd.DataFrame()
        self.load_model()
        unique_sabdab_df = self.load_sabdab_table()
        if self.UNIQUE_PDBIDS:
            unique_sabdab_df = unique_sabdab_df.drop_duplicates(
                subset="pdbid", keep="first"
            ).reset_index()
        if self.PDBS_RANGE:
            unique_sabdab_df = unique_sabdab_df[self.PDBS_RANGE[0] : self.PDBS_RANGE[1]]
        max_i = len(unique_sabdab_df)
        for i, (index, row) in enumerate(unique_sabdab_df.iterrows()):
            if i % 1 == 0:
                cprint(
                    f"building {i} of {max_i}: {index} {row.abid}", color=colors.GREEN
                )
            if i % 50 == 0:
                self.pdbs_df.to_csv(f"{self.PDB_CSV_PATH}.tmp", index=False)
                print(self.pdbs_df)

            for chain_type in self.CHAIN_TYPES:
                try:
                    pdb_df_new = self.build_table_entry(
                        pdbid=row.pdbid, abid=row.abid, chain_type=chain_type
                    )
                    possible_value_vars = pdb_df_new.columns
                    value_vars = [
                        x for x in possible_value_vars if x in pdb_df_new.columns
                    ]
                    pdb_df_new = pd.melt(
                        pdb_df_new,
                        id_vars=["imgt_num"],
                        value_vars=value_vars,
                        var_name="stat",
                        value_name="value",
                    )
                    pdb_df_new["organism"] = row.organism
                    pdb_df_new["pdbid"] = row.pdbid
                    pdb_df_new["abid"] = row.abid
                    pdb_df_new["v_family"] = row.vb
                    pdb_df_new["j_family"] = row.jb
                    pdb_df_new["chain_type"] = chain_type

                    pdb_df_new = pdb_df_new.pivot_table(
                        index=[
                            "organism",
                            "pdbid",
                            "abid",
                            "v_family",
                            "j_family",
                            "chain_type",
                            "stat",
                        ],
                        columns="imgt_num",
                        values="value",
                        aggfunc="first",
                    )
                    self.pdbs_df = pd.concat([self.pdbs_df, pdb_df_new])

                    with open(self.LOG_PATH, "a") as log_file:
                        log_file.write(
                            f"{i},{index},{row.pdbid},{row.abid},{chain_type},success,\n"
                        )
                except Exception as e:
                    with open(self.LOG_PATH, "a") as log_file:
                        err_msg = str(e).replace(",", ";")  # avoid CSV-breaking commas
                        log_file.write(
                            f"{i},{index},{row.pdbid},{row.abid},{chain_type},failure,{err_msg}\n"
                        )
                    cprint(
                        f"[ERROR] Failed on {row.pdbid=}, {row.abid=}, {chain_type=}: {e}",
                        color=colors.RED,
                    )
                    continue

        self.pdbs_df.columns = self.pdbs_df.columns.get_level_values(0)
        column_map = {
            x: f"imgt_{x}"
            for x in self.pdbs_df
            if x.startswith(tuple(list("0123456789")))
        }
        self.pdbs_df = self.pdbs_df.rename(columns=column_map)
        self.sort_columns()
        self.pdbs_df["pdbid"].astype("str")
        self.pdbs_df.to_csv(f"{self.PDB_CSV_PATH}.tmp", index=False)
        self.pdbs_df = pd.read_csv(f"{self.PDB_CSV_PATH}.tmp")
        self.pdbs_df["source"] = [x.split("::")[0] for x in self.pdbs_df["stat"]]
        self.pdbs_df["stat"] = [x.split("::")[1] for x in self.pdbs_df["stat"]]
        self.sort_columns()
        self.pdbs_df = self.pdbs_df.sort_values(
            by=["abid", "chain_type", "source", "stat"]
        )
        os.remove(f"{self.PDB_CSV_PATH}.tmp")
        return self.pdbs_df

    def pdb_get_sequence(self, pdbid, abid):
        pdb_path = self.fetch_pdb(pdbid, self.NUM_SCHEME)
        pdb_Lchains = [abid[len(abid) - 1]]
        pdb_Hchains = [abid[len(abid) - 2]]

        parser = PDBParser()
        structure = parser.get_structure(pdbid, pdb_path)
        model = structure[0]

        # method A: with DSSP
        if self.USE_DSSP_FOR_SEQ:
            rel_dssp = DSSP(model, pdb_path, dssp="mkdssp", acc_array="Wilke")
            dssp_keys = list(rel_dssp.keys())
            filter_keys = dssp_keys
            Hfilter_keys = list(filter(lambda x: x[0] in pdb_Hchains, filter_keys))
            Hfilter_keys = list(
                filter(lambda x: int(x[1][1]) <= self.MAX_NUM_SCHEME, Hfilter_keys)
            )
            Hsequence = "".join([rel_dssp[key][1] for key in Hfilter_keys])
            Lfilter_keys = list(filter(lambda x: x[0] in pdb_Lchains, filter_keys))
            Lfilter_keys = list(
                filter(lambda x: int(x[1][1]) <= self.MAX_NUM_SCHEME, Lfilter_keys)
            )
            Lsequence = "".join([rel_dssp[key][1] for key in Lfilter_keys])
            return Hsequence, Lsequence

        # method B: without DSSP
        if not self.USE_DSSP_FOR_SEQ:
            Hsequence = []
            Lsequence = []
            for chain in model:
                if chain.id in pdb_Hchains:
                    for residue in chain:
                        if PDB.is_aa(residue):
                            Hsequence.append(residue.get_resname())
                if chain.id in pdb_Lchains:
                    for residue in chain:
                        if PDB.is_aa(residue):
                            Lsequence.append(residue.get_resname())
            Hsequence = "".join([seq1(x) for x in Hsequence])
            Lsequence = "".join([seq1(x) for x in Lsequence])
            return Hsequence, Lsequence

        return None, None

    def build_table_entry(self, pdbid, abid, chain_type="H"):
        dfs = {}
        dfs[self.SITECOUNT_NUM_SCHEME] = self._build_table_entry(
            pdbid,
            abid,
            chain_type,
            self.SITECOUNT_NUM_SCHEME,
            self.MAX_NUM_SCHEME,
            add_dnsm=True,
        )
        other_num_schemes = self.ALL_NUM_SCHEMES
        # other_num_schemes = [self.NUM_SCHEME]
        for num_scheme in other_num_schemes:
            if num_scheme == self.SITECOUNT_NUM_SCHEME:
                continue
            dfs[num_scheme] = self._build_table_entry(
                pdbid, abid, chain_type, num_scheme
            )
            dfs[num_scheme] = (
                dfs[num_scheme].iloc[: len(dfs[self.SITECOUNT_NUM_SCHEME])].copy()
            )
        for num_scheme in other_num_schemes:
            if num_scheme in dfs.keys():
                dfs[num_scheme] = dfs[num_scheme].add_prefix(f"{num_scheme}::")

        df = pd.concat(list(dfs.values()))
        df["imgt_num"] = dfs[self.SITECOUNT_NUM_SCHEME][
            f"{self.SITECOUNT_NUM_SCHEME}::num"
        ]
        return df

    def _build_table_entry(
        self, pdbid, abid, chain_type, num_scheme, max_num_scheme=None, add_dnsm=False
    ):
        pdb_path = f"{self.PDB_DB_DIR}/pdb/{num_scheme}/{pdbid}.pdb"
        pdb_Lchains = [abid[len(abid) - 1]]
        pdb_Hchains = [abid[len(abid) - 2]]

        pdb_all_chains = pdb_Lchains + pdb_Hchains
        pdb_chains = []
        if chain_type == "L":
            pdb_chains += pdb_Lchains
        elif chain_type == "H":
            pdb_chains += pdb_Hchains

        # filter and save PDB file
        parser = PDBParser()
        structure = parser.get_structure(pdbid, pdb_path)
        model = structure[0]
        if self.CHAINID_FILTER_METHOD == "new":
            for chain in list(model):
                if chain.id not in pdb_chains:
                    model.detach_child(chain.id)
        temp_pdb_path = f"{pdbid}.temp-{10000 + random.randint(0,9999)}.pdb"
        io = PDBIO()
        io.set_structure(structure)
        io.save(temp_pdb_path)

        # reload filtered PDB file
        structure = parser.get_structure(pdbid, temp_pdb_path)
        model = next(iter(structure))
        dssp = DSSP(model, temp_pdb_path, dssp="mkdssp", acc_array="Wilke")
        os.remove(temp_pdb_path)
        dssp_keys = list(dssp.keys())
        filter_keys = dssp_keys
        filter_keys = list(filter(lambda x: x[0] in pdb_chains, filter_keys))

        # get pdb data
        ca_coords = []
        for key in filter_keys:
            chain_id, res_id = key
            residue = model[chain_id][res_id]
            if "CA" in residue:
                ca_coord = tuple(residue["CA"].coord)
                ca_coords.append(ca_coord)

        # get dssp data
        nums = [f"{x[1][0]}{x[1][1]}{x[1][2]}".replace(" ", "") for x in filter_keys]
        ints = [int(x.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) for x in nums]
        imgt_nums = [f"{num_scheme}_{x}" for x in nums]
        wildtypes = [dssp[key][1] for key in filter_keys]
        rel_asas = [dssp[key][3] for key in filter_keys]

        pdbs_df = pd.DataFrame(
            {
                "int": ints,
                "num": nums,
                "wildtype": wildtypes,
                "rel_asa": rel_asas,
                "ca_coords": ca_coords,
            }
        )

        if max_num_scheme is not None:
            pdbs_df = pdbs_df[pdbs_df["int"] <= max_num_scheme]
        pdbs_df = pdbs_df.drop(["int"], axis=1)
        return pdbs_df

    def build_dnsm_entry(self, abid, chain_type, num_scheme, df=None):
        if df is None:
            df = self.pdbs_df

        new_row = (
            self.get_query(
                df=df,
                query_dict={
                    "abid": abid,
                    "chain_type": chain_type,
                    "source": num_scheme,
                    "stat": "wildtype",
                },
                imgt_cols_only=False,
                dropna=True,
            )
            .iloc[0]
            .copy()
        )

        dnsm_sf = self.get_dnsm(
            df=df, abid=abid, chain_type=chain_type, num_scheme=num_scheme
        )

        imgt_cols = [x for x in new_row.to_frame().T.columns if x.startswith("imgt")]
        update = dict(zip(imgt_cols, dnsm_sf))
        update["stat"] = "dnsm_sf"
        new_row[update.keys()] = list(update.values())
        return new_row.to_frame().T

    def get_non_imgt_cols(self, df=None):
        if df is None:
            df = self.pdbs_df
        non_imgt_cols = [x for x in self.pdbs_df.columns if not x.startswith("imgt")]
        return non_imgt_cols

    def get_imgt_cols(self, df=None, sort=True):
        if df is None:
            df = self.pdbs_df
        imgt_cols = [x for x in self.pdbs_df.columns if x.startswith("imgt")]
        if sort:
            imgt_cols = sorted(imgt_cols, key=self.sort_imgt_nums)
        return imgt_cols

    def get_sorted_columns(self):
        sorted_cols = self.get_non_imgt_cols() + self.get_imgt_cols()
        return sorted_cols

    def sort_columns(self):
        self.pdbs_df = self.pdbs_df.reset_index()
        self.pdbs_df = self.pdbs_df[self.get_sorted_columns()]

    def sort_rows(self):
        self.pdbs_df = self.pdbs_df.sort_values(by=["abid", "chain_type", "stat"])

    def get_query(self, query_dict={}, df=None, dropna=True, imgt_cols_only=True):
        if df is None:
            df = self.pdbs_df
        df_ = df[df[list(query_dict)].eq(pd.Series(query_dict)).all(axis=1)]
        if len(df_) <= 0:
            cprint("[ERROR] get_data results are empty.", color=colors.RED)
            return None
        if imgt_cols_only:
            df_ = df_[self.get_imgt_cols()]
        if dropna:
            df_ = df_.dropna(how="all", axis=1)
        return df_

    def get_seq(self, abid, chain_type, num_scheme, df=None):
        if df is None:
            df = self.pdbs_df
        seq_row = self.get_query(
            query_dict={
                "abid": abid,
                "chain_type": chain_type,
                "source": num_scheme,
                "stat": "wildtype",
            },
            imgt_cols_only=True,
            dropna=True,
        )
        if not (len(seq_row) == 1):
            print(f"[ERROR] {abid}\n{seq_row}")
            raise Exception(f"[ERROR] wrong number of rows: {abid} {len(seq_row)}")
        return seq_row

    def get_dnsm(self, abid, chain_type, num_scheme, df=None):
        if df is None:
            df = self.pdbs_df
        seq_df = self.get_seq(abid, chain_type, num_scheme, df)
        aa_seq = "".join(list(seq_df.iloc[0]))
        dnsm_sf = self.get_selection_factors(aa_seq, self.dnsm_model, model_type="dnsm")
        assert len(dnsm_sf) == len(aa_seq)
        return dnsm_sf

    # dms table utils

    @staticmethod
    def get_selection_factors(aa_seq, model, model_type="dnsm"):
        if model_type == "dnsm":
            result = model.selection_factors_of_aa_str([aa_seq, ""])
            selection_factors = result[0].detach().numpy()
        elif model_type == "dasm":
            result = model.selection_factors_of_aa_str([aa_seq, ""])
            selection_factors = result[0].detach().numpy()
            selection_factors = np.nan_to_num(selection_factors, nan=0.0)

        return selection_factors

    @staticmethod
    def imgt_label_to_number(imgt_label):
        imgt_number = int(
            float(imgt_label.replace("imgt_", "").strip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        )
        return imgt_number

    @staticmethod
    def imgt_alpha_to_decimal(imgt_alpha_cols):
        imgt_dec_cols = [
            (
                f"{x[:-1]}.{ord(x[-1]) - 64}"
                if x[-1] in ("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                else x
            )
            for x in imgt_alpha_cols
        ]
        return imgt_dec_cols

    @staticmethod
    def sort_imgt_nums(col):
        number_part = DmsViz_Utility.imgt_label_to_number(col)
        suffix_part = ord(col[-1])
        # reverse ordering of site 112
        if number_part == 112:
            return (number_part, -suffix_part)
        return (number_part, suffix_part)

    @staticmethod
    def get_common_imgt_cols(imgt_cols):
        common_imgt_cols = [
            x for x in imgt_cols if x[-1] not in ("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        ]
        return common_imgt_cols

    @staticmethod
    def get_uncommon_imgt_cols(imgt_cols):
        uncommon_imgt_cols = [
            x for x in imgt_cols if x[-1] in ("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        ]
        return uncommon_imgt_cols

    @staticmethod
    def get_color_palette(num_colors, palette_name="husl"):
        colors = sns.color_palette(palette_name, num_colors)
        colors = colors.as_hex()
        return colors


class DmsViz_Table:
    CDR_REGIONS = DmsViz_Utility.CDR_REGIONS
    ROUND_DIGITS = 4

    def __init__(self, pdbs_df, do_init=True):
        self.df = pdbs_df
        if do_init:
            self.init()

    def init(self):
        self.update_cols_ordering()
        self.imgt_nums = self.get_imgt_nums(self.imgt_cols)
        self.padded_imgt_nums = self.get_padded_imgt_nums(
            self.imgt_cols, self.imgt_nums
        )
        self.cdr_region = self.get_cdr_region_cols()
        self.is_cdr_region = [(x is not None) for x in self.cdr_region]
        self.df = self.df[self.header_cols + self.imgt_cols]

    def update_cols_ordering(self):
        self.header_cols = self.get_header_cols()
        self.imgt_cols = DmsViz_Table.get_imgt_cols(self.df.columns, is_sorted=True)
        self.df = self.df[self.header_cols + self.imgt_cols]

    def get_header_cols(self):
        imgt_cols = [x for x in self.df.columns if not x.startswith("imgt")]
        return imgt_cols

    @staticmethod
    def get_imgt_cols(df_cols, is_sorted=True, include_prefix=True):
        imgt_cols = [x for x in df_cols if x.startswith("imgt")]
        if is_sorted:
            imgt_cols = sorted(imgt_cols, key=DmsViz_Utility.sort_imgt_nums)
        if not include_prefix:
            imgt_cols = [x.replace("imgt_", "") for x in imgt_cols]
        return imgt_cols

    @staticmethod
    def get_imgt_nums(imgt_cols):
        imgt_nums = [DmsViz_Utility.imgt_label_to_number(x) for x in imgt_cols]
        return imgt_nums

    @staticmethod
    def get_padded_imgt_nums(imgt_cols, imgt_nums):
        num_ins = 0
        prv_imgt_col = ""
        imgt_nums_padded = []
        for i, (imgt_num, imgt_col) in enumerate(zip(imgt_nums, imgt_cols)):
            stripped_imgt_col = imgt_col.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            if stripped_imgt_col == prv_imgt_col:
                num_ins += 1
            imgt_nums_padded.append(imgt_num + num_ins)
            prv_imgt_col = stripped_imgt_col

        return imgt_nums_padded

    def get_cdr_region_cols(self):
        cdr_region_cols = [None] * len(self.imgt_cols)
        for i, imgt_num in enumerate(self.imgt_nums):
            for cdr_label, cdr_range in self.CDR_REGIONS.items():
                if imgt_num >= cdr_range[0] and imgt_num <= cdr_range[1]:
                    cdr_region_cols[i] = cdr_label
        return cdr_region_cols

    def get_cdr_cols_dict(self):
        cdr_cols_dict = {}
        for cdr, cdr_range in self.CDR_REGIONS.items():
            indices = list(filter(lambda x: x[1] == cdr, enumerate(self.cdr_region)))
            cdr_cols_dict[cdr] = [self.imgt_cols[i[0]] for i in indices]
        return cdr_cols_dict

    def get_cdr_length_counts(self, add_to_df=True):
        cdr_cols = self.get_cdr_cols_dict()
        cdr_length_dict = {}
        for cdr in self.CDR_REGIONS.keys():
            cdr_name = f"{cdr}_len"
            cdr_length_dict[cdr_name] = []
            for index, row in self.df.iterrows():
                length = row[cdr_cols[cdr]].count()
                cdr_length_dict[cdr_name].append(length)
        if add_to_df:
            self.df = self.df.assign(**cdr_length_dict)
            self.update_cols_ordering()
        return cdr_length_dict

    def split_by_stat(self, include_headers, metrics=["rel_asa", "dnsm_sf"]):
        split_dfs = {}
        seq_df = self.df[self.df.stat == "wildtype"]
        seq_df[self.imgt_cols] = (
            seq_df[self.imgt_cols].astype(str).replace("nan", pd.NA)
        )
        split_dfs["wildtype"] = seq_df

        for stat_type in set(self.df.stat):
            if stat_type == "wildtype":
                continue
            if metrics is None or stat_type not in metrics:
                continue
            stat_df = (
                self.df[self.df.stat == stat_type].reset_index().drop("index", axis=1)
            )
            stat_df[self.imgt_cols] = stat_df[self.imgt_cols].astype(float)
            split_dfs[stat_type] = stat_df

        if not include_headers:
            for stat_type in split_dfs.keys():
                split_dfs[stat_type] = split_dfs[stat_type][self.imgt_cols]
        return split_dfs

    def stat_to_list(self, abid, stat, dropna=True):
        df = self.df[(self.df.abid == abid) & (self.df.stat == stat)]
        df = df[self.imgt_cols]
        if dropna:
            df = df.dropna(axis=1, how="all")

        return df.iloc[0].tolist()

    def build_stat_df(self, metrics=["rel_asa", "dnsm_sf"]):
        split_dfs = self.split_by_stat(include_headers=False, metrics=metrics)
        stat_df = pd.DataFrame()
        stat_df["imgt"] = self.imgt_cols
        stat_df["imgt_num"] = self.imgt_nums
        stat_df["imgt_label"] = [x.replace("imgt_", "") for x in self.imgt_cols]
        stat_df["imgt_occurrences"] = list(split_dfs["dnsm_sf"].count())
        stat_df["cdr_region"] = self.cdr_region
        stat_df["is_cdr_region"] = self.is_cdr_region
        stat_df["aa_mode"] = list(split_dfs["wildtype"].mode(dropna=True).iloc[0])
        stat_df["aa_mode_count"] = [
            (split_dfs["wildtype"][imgt_col] == aa_mode).sum()
            for imgt_col, aa_mode in zip(stat_df.imgt, stat_df.aa_mode)
        ]
        stat_df["aa_mode_perc"] = stat_df.aa_mode_count / stat_df.imgt_occurrences
        for stat_type in split_dfs.keys():
            if stat_type == "wildtype":
                continue
            stat_df[f"{stat_type}_median"] = list(
                split_dfs[stat_type].median(skipna=True).round(self.ROUND_DIGITS)
            )
            stat_df[f"{stat_type}_mean"] = list(
                split_dfs[stat_type].mean(skipna=True).round(self.ROUND_DIGITS)
            )
            q_df = (
                split_dfs[stat_type]
                .quantile(np.linspace(0.0, 1.0, 4 + 1))
                .round(self.ROUND_DIGITS)
            )
            q_df = [tuple(x) for x_name, x in q_df.items()]
            stat_df[f"{stat_type}_quantiles"] = q_df
        return stat_df

    def build_aligned_df(self, abid, dropna=True, metrics=["rel_asa", "dnsm_sf"]):
        df = self.df[self.df.abid == abid]
        df = df[df.stat.isin(tuple(["wildtype"] + metrics))]
        align_df = pd.DataFrame()
        align_df["imgt_col"] = [x for x in df.columns if x.startswith("imgt")]
        align_df["imgt_num"] = [
            DmsViz_Utility.imgt_label_to_number(x) for x in align_df.imgt_col
        ]
        align_df["wildtype"] = (
            df[df.stat == "wildtype"][align_df["imgt_col"]].iloc[0].to_list()
        )
        for stat_type in set(df.stat):
            if stat_type == "wildtype":
                continue
            align_df[stat_type] = (
                df[df.stat == stat_type][align_df["imgt_col"]]
                .iloc[0]
                .astype("float")
                .to_list()
            )
        align_df = align_df[align_df.imgt_num <= DmsViz_Utility.MAX_NUM_SCHEME]
        align_df = align_df.sort_values(
            by="imgt_col", key=lambda x: x.map(DmsViz_Utility.sort_imgt_nums)
        )

        if dropna:
            align_df.dropna(axis=0, how="any")
        return align_df

    def build_aligned_diff_df(self, v3_abid, v4_abid, dropna=True):
        abid_dict = {"v3": v3_abid, "v4": v4_abid}
        aa_seqs = []
        dnsm_sfs = []

        for name, abid in abid_dict.items():
            aa_seq = self.stat_to_list(abid, "wildtype", dropna=False)
            dnsm_sf = self.stat_to_list(abid, "dnsm_sf", dropna=False)
            dnsm_sf = [float(x) for x in dnsm_sf]
            aa_seqs.append(aa_seq)
            dnsm_sfs.append(dnsm_sf)

        # align_dict common columns between v3 and v4
        align_dict = {}
        dnsm_diff = [x - y for x, y in zip(dnsm_sfs[0], dnsm_sfs[1])]
        align_dict["imgt_col"] = [
            x for x, y in zip(self.imgt_cols, dnsm_diff) if not math.isnan(y)
        ]
        align_dict["imgt_num"] = [
            x for x, y in zip(self.imgt_nums, dnsm_diff) if not math.isnan(y)
        ]
        dropped_imgt_cols = [
            x for x, y in zip(self.imgt_cols, dnsm_diff) if math.isnan(y)
        ]
        align_dict["v3_seq"] = [
            x for x, y in zip(aa_seqs[0], dnsm_diff) if not math.isnan(y)
        ]
        align_dict["v4_seq"] = [
            x for x, y in zip(aa_seqs[1], dnsm_diff) if not math.isnan(y)
        ]
        align_dict["v3v4_seq"] = [
            x if (x == y) else "-"
            for x, y in zip(align_dict["v3_seq"], align_dict["v4_seq"])
        ]
        align_dict["v3v4_mut"] = [
            "-" if (x == y) else y
            for x, y in zip(align_dict["v3_seq"], align_dict["v4_seq"])
        ]
        align_dict["v3_dnsm"] = [
            x for x, y in zip(dnsm_sfs[0], dnsm_diff) if not math.isnan(y)
        ]
        align_dict["v4_dnsm"] = [
            x for x, y in zip(dnsm_sfs[1], dnsm_diff) if not math.isnan(y)
        ]
        align_dict["dnsm_diff"] = dnsm_diff
        align_dict["dnsm_diff"] = [x for x in dnsm_diff if not math.isnan(x)]
        align_df = pd.DataFrame(align_dict)
        align_df = align_df[align_df.imgt_num <= DmsViz_Utility.MAX_NUM_SCHEME]

        if dropna:
            align_df.dropna(axis=0, how="any")
        return align_df

    @staticmethod
    def append_data_to_aligned_df(
        align_df,
        data_df,
        data_metrics,
        align_relabel,
        data_relabel,
        merge_how="inner",
    ):
        align_df = align_df.rename(columns=align_relabel)
        data_df = data_df.rename(columns=data_relabel)
        data_df = data_df[["imgt_col"] + data_metrics]
        align_df = pd.merge(align_df, data_df, on="imgt_col", how=merge_how)
        return align_df

    def get_mean_stats_of_pdbids(
        self, pdbids, name, dropna=True, use_mode_for_wildtype=False
    ):
        mean = {}
        df = self.df[self.df.pdbid.isin(pdbids)]
        df_table = DmsViz_Table(df)
        split_dfs = df_table.split_by_stat(include_headers=False)
        mean["imgt_col"] = df_table.imgt_cols

        mean["wildtype"] = split_dfs["wildtype"].mode().iloc[0].tolist()
        for stat_type in split_dfs.keys():
            if stat_type == "wildtype":
                continue
            mean[stat_type] = split_dfs[stat_type].mean().to_list()

        mean["wildtype"] = split_dfs["wildtype"].iloc[0]
        if not use_mode_for_wildtype:
            for i, row in split_dfs["wildtype"].iterrows():
                mean["wildtype"] = [
                    x if (x == y) else "-" for x, y in zip(mean["wildtype"], row)
                ]

        mean_df = pd.DataFrame(mean).set_index("imgt_col").T
        mean_df.index.name = "stat"
        mean_df = mean_df.reset_index()
        mean_df.insert(0, "abid", [name] * len(split_dfs))
        mean_df.insert(0, "pdbid", [name] * len(split_dfs))
        return mean_df

    def make_dnsm_viz_json(
        self,
        abid,
        output_path,
        temp_dir,
        metrics={"DNSM": "dnsm_sf"},
    ):
        pdbid, h_chainid, l_chainid = DmsViz.abid_get_pdbid(abid)
        aa_seq = self.stat_to_list(abid, "wildtype")

        metric_names, metric_cols = list(metrics.keys()), list(metrics.values())
        metric_name = metric_names[0]
        metric_col = metric_cols[0]
        metric_data = self.stat_to_list(abid, metric_col)
        metric_df = DmsViz.write_metric_csv(
            aa_seq,
            metric_data,
            metric_name,
            f"{temp_dir}/{abid}.metric.{metric_col}.csv",
        )
        sitemap_df = DmsViz.write_sitemap_csv(
            len(metric_df), f"{temp_dir}/{abid}.sitemap.csv"
        )

        included_chains = [h_chainid]
        excluded_chains = list("ABCDEFHIJKLMNOPQRSTUVWXYZ")
        excluded_chains.remove(h_chainid)
        if l_chainid in excluded_chains:
            excluded_chains.remove(l_chainid)

        DmsViz.write_dms_viz_json(
            name=f"{pdbid}",
            pdbid=pdbid,
            plot_colors=DmsViz.COLORS,
            metric=metric_name,
            included_chains=included_chains,
            excluded_chains=excluded_chains,
            input_metric_csv_path=f"{temp_dir}/{abid}.metric.{metric_col}.csv",
            input_sitemap_csv_path=f"{temp_dir}/{abid}.sitemap.csv",
            output_json_path=f"{output_path}",
        )
        return metric_df

    def make_diff_viz_json(
        self,
        structure_abid,
        v3_name,
        v4_name,
        output_path,
        temp_dir,
    ):
        align_df = self.build_aligned_diff_df(v3_name, v4_name)

        diff_df = DmsViz.write_metric_csv(
            aa_seq=align_df["v3_seq"],
            metric_data=align_df["dnsm_diff"],
            metric_name="DNSM_Diff",
            output_path=f"{temp_dir}/{v3_name}_{v4_name}.dnsm_diff.csv",
            mutant=align_df["v3v4_mut"],
        )
        v3_df = DmsViz.write_metric_csv(
            aa_seq=align_df["v3_seq"],
            metric_data=align_df["v3_dnsm"],
            metric_name="DNSM_V3",
            output_path=f"{temp_dir}/{v3_name}_{v4_name}.dnsm_v3.csv",
            mutant=align_df["v3v4_mut"],
        )
        v4_df = DmsViz.write_metric_csv(
            aa_seq=align_df["v3_seq"],
            metric_data=align_df["v4_dnsm"],
            metric_name="DNSM_V4",
            output_path=f"{temp_dir}/{v3_name}_{v4_name}.dnsm_v4.csv",
            mutant=align_df["v3v4_mut"],
        )
        sitemap_df = DmsViz.write_sitemap_csv(
            site_count=len(diff_df),
            output_path=f"{temp_dir}/{v3_name}_{v4_name}.sitemap.csv",
        )
        combined_df = DmsViz.write_combined_csv(
            metric_csv_paths=[
                f"{temp_dir}/{v3_name}_{v4_name}.dnsm_diff.csv",
                f"{temp_dir}/{v3_name}_{v4_name}.dnsm_v3.csv",
                f"{temp_dir}/{v3_name}_{v4_name}.dnsm_v4.csv",
            ],
            metric_names=["DNSM_Diff", "DNSM_V3", "DNSM_V4"],
            output_path=f"{temp_dir}/{v3_name}_{v4_name}.combined.csv",
        )

        pdbid, h_chainid, l_chainid = DmsViz.abid_get_pdbid(structure_abid)
        included_chains = [h_chainid]
        excluded_chains = list("ABCDEFHIJKLMNOPQRSTUVWXYZ")
        excluded_chains.remove(h_chainid)
        if l_chainid in excluded_chains:
            excluded_chains.remove(l_chainid)

        # add condition
        add_options = ""
        condition_options = '--condition "condition" '
        condition_options += '--condition-name "Selection Factor" '
        add_options += condition_options

        DmsViz.write_dms_viz_json(
            name=f"V3/V4 Comparison: {v3_name} vs {v4_name}",
            pdbid=pdbid,
            plot_colors=DmsViz_Utility.get_color_palette(3),
            metric="factor",
            included_chains=included_chains,
            excluded_chains=excluded_chains,
            # input_metric_csv_path=f'{temp_dir}/{v3_name}_{v4_name}.dnsm_diff.csv',
            input_metric_csv_path=f"{temp_dir}/{v3_name}_{v4_name}.combined.csv",
            input_sitemap_csv_path=f"{temp_dir}/{v3_name}_{v4_name}.sitemap.csv",
            output_json_path=f"{output_path}",
            add_options=add_options,
        )

        return pd.DataFrame(align_df)

    @staticmethod
    def make_viz_json(
        title,
        name,
        structure_abid,
        align_df,
        seq_col,
        mut_col,
        metric_cols,
        metric_names,
        output_path,
        temp_dir,
    ):
        metric_dfs = []
        metric_csv_paths = []
        for metric_col, metric_name in zip(metric_cols, metric_names):
            metric_csv_path = f"{temp_dir}/{name}.{metric_col}.csv"
            mutant = "-"
            if mut_col:
                mutant = align_df[mut_col]
            metric_df = DmsViz.write_metric_csv(
                align_df[seq_col],
                align_df[metric_col],
                metric_name,
                metric_csv_path,
                mutant=mutant,
            )
            metric_dfs.append(metric_df)
            metric_csv_paths.append(metric_csv_path)

        sitemap_df = DmsViz.write_sitemap_csv(
            len(metric_dfs[0]), f"{temp_dir}/{name}.sitemap.csv"
        )
        combined_df = DmsViz.write_combined_csv(
            metric_csv_paths=metric_csv_paths,
            metric_names=metric_names,
            output_path=f"{temp_dir}/{name}.combined.csv",
        )

        pdbid, h_chainid, l_chainid = DmsViz.abid_get_pdbid(structure_abid)
        included_chains = [h_chainid]
        excluded_chains = list("ABCDEFHIJKLMNOPQRSTUVWXYZ")
        excluded_chains.remove(h_chainid)
        excluded_chains.remove(l_chainid)

        # add condition
        add_options = ""
        condition_options = '--condition "condition" '
        condition_options += '--condition-name "Selection Factor" '
        add_options += condition_options

        DmsViz.write_dms_viz_json(
            name=f"{title}",
            pdbid=pdbid,
            plot_colors=DmsViz_Utility.get_color_palette(len(metric_cols)),
            metric="factor",
            included_chains=included_chains,
            excluded_chains=excluded_chains,
            input_metric_csv_path=f"{temp_dir}/{name}.combined.csv",
            input_sitemap_csv_path=f"{temp_dir}/{name}.sitemap.csv",
            output_json_path=f"{output_path}",
            add_options=add_options,
        )
        return align_df


class DmsViz:
    USE_LOCAL_PDB_STRUCTURE = False
    OUTPUT_DIR = "_temp"
    COLORS = ["#675ed6", "#808080"]

    def __init__(self, output_dir):
        self.output_dir = output_dir

    @staticmethod
    def abid_get_pdbid(abid):
        pdbid = abid[:-2]
        h_chainid = abid[len(abid) - 2]
        l_chainid = abid[len(abid) - 1]
        return pdbid, h_chainid, l_chainid

    @staticmethod
    def run_command(command, do_print=False):
        if do_print:
            cprint(f"COMMAND: {command}")
            os.environ["LAST_COMMAND"] = command
        try:
            output = subprocess.run(
                command, shell=True, check=True, capture_output=True, text=True
            )
        except Exception as e:
            cprint(f"COMMAND failed with exception: {e}", color=colors.RED)
            return None
        else:
            cprint(f"COMMAND successful!", color=colors.GREEN)
            return output

    @staticmethod
    def copy_file(src, dest, use_symlink=False, use_move=False):
        if use_symlink:
            relative_src = os.path.relpath(src, start=os.path.dirname(dest))
            os.symlink(relative_src, dest)
            cprint(f"File symlinked: {src} -> {dest} ({relative_src})")
        if use_move:
            shutil.move(src, dest)
            cprint(f"File moved: {src} -> {dest}")
        else:
            shutil.copy(src, dest)
            cprint(f"File copied: {src} -> {dest}")

    @staticmethod
    def write_csv(df, output_path, add_index=False):
        df.to_csv(output_path, index=add_index)
        cprint(f'Outputting csv: "{output_path}"', color=colors.YELLOW)

    @staticmethod
    def write_sitemap_csv(site_count, output_path=None):
        sitemap_df = pd.DataFrame(
            {
                "sequential_site": range(1, site_count + 1),
                "reference_site": range(1, site_count + 1),
            }
        )

        if output_path is not None:
            DmsViz.write_csv(sitemap_df, output_path, False)
        return sitemap_df

    @staticmethod
    def write_metric_csv(
        aa_seq, metric_data, metric_name, output_path=None, mutant="-"
    ):
        dnsm_df = pd.DataFrame({"wildtype": list(aa_seq)})
        dnsm_df["site"] = dnsm_df.index + 1
        dnsm_df["mutant"] = mutant
        dnsm_df[metric_name] = list(metric_data)

        if output_path is not None:
            DmsViz.write_csv(dnsm_df, output_path, False)
        return dnsm_df

    @staticmethod
    def write_combined_csv(metric_csv_paths, metric_names, output_path=None):
        combined_df = pd.DataFrame()
        for i, (metric_csv_path, metric_name) in enumerate(
            zip(metric_csv_paths, metric_names)
        ):
            metric_df = pd.read_csv(metric_csv_path)
            if i == 0:
                combined_df[["wildtype", "site", "mutant"]] = metric_df[
                    ["wildtype", "site", "mutant"]
                ]
            combined_df[metric_name] = metric_df[metric_name]

        combined_df = combined_df.melt(
            id_vars=["wildtype", "site", "mutant"],
            value_vars=metric_names,
            var_name="condition",
            value_name="factor",
        )

        if output_path is not None:
            DmsViz.write_csv(combined_df, output_path, False)
        return combined_df

    @staticmethod
    def write_dms_viz_json(
        name,
        pdbid,
        plot_colors,
        metric,
        included_chains,
        excluded_chains,
        input_metric_csv_path,
        input_sitemap_csv_path,
        output_json_path,
        add_options="",
        local_pdb_path=None,
    ):
        try:
            if not os.path.isfile(input_metric_csv_path):
                raise Exception(
                    f"input_metric_csv_path does not exist: {input_metric_csv_path}"
                )
            if not os.path.isfile(input_sitemap_csv_path):
                raise Exception(
                    f"input_sitemap_csv_path does not exist: {input_sitemap_csv_path}"
                )
        except Exception as e:
            cprint(f"run_config_dms_viz failed: {e}", color=colors.RED)

        prog = f"configure-dms-viz format"
        plot_colors_str = ",".join(plot_colors)
        base_options = f'--name "{name}" \
                        --title "{name}" \
                        --description "SABDAB: {name}" \
                        --colors "{plot_colors_str}" \
                        --metric "{metric}" '
        my_pdbid = pdbid
        if local_pdb_path and DmsViz.USE_LOCAL_PDB_STRUCTURE:
            my_pdbid = local_pdb_path
        base_options += f'--structure "{my_pdbid}" '
        if len(included_chains) != 0:
            base_options += f'--included-chains "{" ".join(included_chains)}" '
        if len(excluded_chains) != 0:
            base_options += f'--excluded-chains "{" ".join(excluded_chains)}" '

        cmd = f'{prog} {base_options} \
                --input "{input_metric_csv_path}" \
                --sitemap "{input_sitemap_csv_path}" \
                --output "{output_json_path}" \
                {add_options} '

        cmd = " ".join(cmd.split())
        output = DmsViz.run_command(cmd)

        if output is None:
            raise Exception("ERROR: configure-dms-viz format run failed.")
        else:
            cprint(f'Outputting json: "{output_json_path}"', color=colors.YELLOW)
        return output


class DmsViz_DatabaseManager:
    db_sources = ["rcsb", "opig-chothia", "opig-imgt"]
    pdb_url_path_dict = {
        "rcsb": "https://files.rcsb.org/view/{pdbid}.pdb",
        "opig-chothia": "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdbid}/?scheme=chothia",
        "opig-imgt": "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdbid}/?scheme=imgt",
    }
    fasta_url_path_dict = {
        "rcsb": "https://www.rcsb.org/fasta/entry/{pdb_id}/display",
    }

    @staticmethod
    def download_file(url_path, file_path):
        if not os.path.exists(file_path):
            try:
                result = subprocess.run(
                    ["wget", "-O", file_path, url_path],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"File downloaded successfully: {url_path} -> {file_path}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download file: {e}")
                print(f"Output: {e.output}")
        else:
            print(f"File already exists: {file_path}")

    def __init__(self, db_root_path="pdb-db", default_db_source="rcsb"):
        self.db_path = db_root_path
        self.default_db_source = default_db_source

    def init_db(self):
        os.makedirs(f"{self.db_path}", exist_ok=True)
        for db_source in self.db_sources:
            os.makedirs(f"{self.db_path}/pdb/{db_source}", exist_ok=True)

    # pdb file access

    def pdb_get_url_path(self, pdbid, db_source):
        if db_source in self.pdb_url_path_dict.keys():
            url_path = self.pdb_url_path_dict[db_source].format(pdbid=pdbid)
        else:
            raise TypeError(f"Unrecognized db_source type: {db_source}")
        return url_path

    def pdb_get_db_path(self, pdbid, db_source):
        db_path = f"{self.db_path}/pdb/{db_source}/{pdbid}.pdb"
        return db_path

    @staticmethod
    def get_db_source(self, db_source=None):
        if db_source is None:
            return self.default_db_source
        return db_source

    def pdb_download(self, pdbid, db_source=None):
        url_path = self.pdb_get_url_path(pdbid, db_source)
        file_path = self.pdb_get_db_path(pdbid, db_source)
        self.download_file(url_path, file_path)

    def pdb_fetch(self, pdbid, db_source=None):
        db_path = self.pdb_get_db_path(pdbid, db_source)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"db_path does not exist: {db_path}")
        return db_path

    def pdb_fetch_or_download(self, pdbid, db_source=None):
        db_source = self.get_db_source(db_source)
        db_path = self.pdb_get_db_path(pdbid, db_source)
        if not os.path.exists(db_path):
            self.pdbid_download()
        return self.pdb_fetch(self, pdbid, db_source)

    # fasta file access

    def fasta_get_url_path(self, pdbid, db_source="rcsb"):
        if db_source in self.fasta_url_path_dict.keys():
            url_path = self.fasta_url_path_dict[db_source].format(pdbid=pdbid)
        else:
            raise TypeError(f"Unrecognized db_source type: {db_source}")
        return url_path


### MAIN ###


# default input paths
SCRIPT_DIR = f"{os.getcwd()}"
SHARED_DIR = "/fh/fast/matsen_e/shared/bcr-mut-sel"
SABDAB_DIR = f"{SHARED_DIR}/sabdab"
SABDAB_PATH = f"{SABDAB_DIR}/sabdab_summary_for_dnsm.tsv"
PDBS_CSV_PATH = f"{SABDAB_DIR}/pdb-db.ALL.csv"
PDB_DB_DIR = f"{SABDAB_DIR}/pdb-db"
DNSM_MODELS_DIR = f"{SHARED_DIR}/dnsm/dnsm-experiments-1/dnsm-train/trained_models"
DASM_MODELS_DIR = f"{SHARED_DIR}/dasm/dnsm-experiments-1/dasm-train/trained_models"
DNSM_MODEL_NAME = "dnsm_1m-v1jaffe+v1tang-joint"
# default output paths
OUTPUT_DIR = f"{SABDAB_DIR}/_output"
TEMP_DIR = f"{SABDAB_DIR}/_temp"
LOG_PATH = f"{TEMP_DIR}/build_pdb_table.log"


def pdb_csv_build(
    output_dir=OUTPUT_DIR,
    temp_dir=TEMP_DIR,
    sabdab_path=SABDAB_PATH,
    pdb_db_dir=PDB_DB_DIR,
    dnsm_models_dir=DNSM_MODELS_DIR,
    dnsm_model_name=DNSM_MODEL_NAME,
    pdbs_range=None,
    log_path=LOG_PATH,
    always_rebuild=False,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    util = DmsViz_Utility(
        output_dir=output_dir,
        sabdab_path=sabdab_path,
        pdb_db_dir=pdb_db_dir,
        dnsm_models_dir=dnsm_models_dir,
        dnsm_model_name=dnsm_model_name,
    )
    util.ALWAYS_REBUILD_CSV = always_rebuild
    util.PDBS_RANGE = pdbs_range
    util.LOG_PATH = log_path
    util.update_pdbs_csv_path()
    pdbs_df = util.save_pdbs_table()
    display(pdbs_df)


def pdb_csv_add_dnsm(
    output_dir=OUTPUT_DIR,
    temp_dir=TEMP_DIR,
    sabdab_path=SABDAB_PATH,
    pdb_db_dir=PDB_DB_DIR,
    dnsm_models_dir=DNSM_MODELS_DIR,
    dnsm_model_name=DNSM_MODEL_NAME,
    pdbs_range=None,
    pdb_scheme=None,
    chain_type=None,
):
    successes, errors = [], []
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    path = f"pdb-db.{DNSM_MODEL_NAME}.dnsm_sf.csv"
    if pdbs_range is not None:
        path = path.replace(".csv", f".{pdbs_range[0]}-{pdbs_range[1]}.csv")
    else:
        path = path.replace(".csv", ".ALL.csv")
    temp_path = f"{TEMP_DIR}/{path}.tmp"
    output_path = f"{OUTPUT_DIR}/{path}"

    util = DmsViz_Utility(
        output_dir=output_dir,
        temp_dir=temp_dir,
        sabdab_path=sabdab_path,
        pdb_db_dir=pdb_db_dir,
        dnsm_models_dir=dnsm_models_dir,
        dnsm_model_name=dnsm_model_name,
    )
    util.load_model()
    pdbs_df = util.load_pdbs_table()

    query_dict = {
        "stat": "wildtype",
        "chain_type": chain_type,
        "source": pdb_scheme,
    }
    query_dict = dict_remove_empty(query_dict)
    print(f"{query_dict=}")
    pdbs_df = util.get_query(
        df=pdbs_df,
        query_dict=query_dict,
        imgt_cols_only=False,
        dropna=False,
    ).reset_index(drop=True)
    if pdbs_range is not None:
        pdbs_df = pdbs_df[pdbs_range[0] : pdbs_range[1]]
    display(pdbs_df)

    new_rows = []
    dnsm_df = pd.DataFrame({}, columns=util.pdbs_df.columns)
    for i, (id, row) in enumerate(pdbs_df.iterrows()):
        cprint(
            f"building dnsm {i} of {len(pdbs_df)}: {id} {row.abid} {row.stat} {row.source}"
        )
        try:
            new_row = util.build_dnsm_entry(
                abid=row.abid, chain_type=row.chain_type, num_scheme=row.source
            )
            new_rows.append(new_row)
        except Exception as e:
            cprint(
                f"[ERROR] error occurred during: {i}={id} {row.abid}: {e}",
                color=colors.RED,
            )
            errors.append(row.abid)
        else:
            cprint(f"[SUCCESS] during: {i}={id} {row.abid}", color=colors.GREEN)
            successes.append(row.abid)
        if i % 100 == 0:
            cprint(f"saving to temp_path: {temp_path}", color=colors.GREEN)
            dnsm_df = pd.concat(new_rows, ignore_index=True)
            display(dnsm_df)
            dnsm_df.to_csv(temp_path, index=False)

    cprint(f"[ERROR] total_errors: {len(errors)} {errors}", color=colors.RED)
    cprint(f"saving to output_path: {output_path}", color=colors.GREEN)
    dnsm_df = pd.concat(new_rows, ignore_index=True)
    display(dnsm_df)
    dnsm_df.to_csv(output_path)


def pdb_csvs_join(input_paths, output_path):
    dfs = {}
    for csv_path in input_paths:
        print(f"csv_path: {csv_path}")
        dfs[csv_path] = pd.read_csv(csv_path)
        print(f"csv_path: {csv_path}\n{dfs[csv_path]}")
    df = pd.concat(list(dfs.values()))
    cprint(f"saving to output_path: {output_path}", color=colors.GREEN)
    display(df)
    df.to_csv(output_path, index=False)


def parse_args():
    arg_parser = argparse.ArgumentParser("Build sabdab sequence data table.")
    arg_parser.add_argument("--pdb-csv-build-db", action="store_true", default=None)
    arg_parser.add_argument("--pdb-csv-build-dnsm", action="store_true", default=None)
    arg_parser.add_argument("--pdb-csvs-join", action="store_true", default=None)

    arg_parser.add_argument("--temp-dir", type=str, default="_temp")
    arg_parser.add_argument("--output-dir", type=str, default="_output")
    arg_parser.add_argument("--input-csvs", type=str, nargs="+", default=None)
    arg_parser.add_argument("--sabdab", type=str, default=SABDAB_PATH)
    arg_parser.add_argument("--range", type=int, nargs=2, default=None)
    arg_parser.add_argument("--rebuild", action="store_true", default=False)
    arg_parser.add_argument("--log", type=str, default=LOG_PATH)
    arg_parser.add_argument("--pdb-scheme", type=str, default=None)
    arg_parser.add_argument("--chain-type", type=str, default=None)
    arg_parser.add_argument("--dnsm-models_dir", type=str, default=DNSM_MODELS_DIR)
    arg_parser.add_argument("--dnsm-model-name", type=str, default=DNSM_MODEL_NAME)

    args = arg_parser.parse_args()
    return args.__dict__


def main():
    args = parse_args()
    print(args)

    if args["pdb_csv_build_db"]:
        print("# building pdb_csv...")
        pdb_csv_build(
            pdbs_range=args["range"],
            sabdab_path=args["sabdab"],
            log_path=args["log"],
            always_rebuild=args["rebuild"],
        )

    elif args["pdb_csv_build_dnsm"]:
        print("# adding dnsm to pdb_csv...")
        pdb_csv_add_dnsm(
            pdbs_range=args["range"],
            sabdab_path=args["sabdab"],
            pdb_scheme=args["pdb_scheme"],
            chain_type=args["chain_type"],
        )

    elif args["pdb_csvs_join"]:
        print("# join pdb_csvs...")
        pdb_csvs_join(
            input_paths=args["input_csvs"],
            output_path=args["output"],
        )


if __name__ == "__main__":
    main()
