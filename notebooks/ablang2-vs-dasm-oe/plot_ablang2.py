import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from netam.oe_plot import plot_sites_observed_vs_expected
from dnsmex.local import localify

figures_dir = localify("FIGURES_DIR")

with open(f'_ignore/rodriguez_numbering.pkl', 'rb') as f:
    numbering = pickle.load(f)
    
fig, ax = plt.subplots(figsize=[15,5])
fig.patch.set_facecolor('white')

# These files are in /home/matsen/archive/2025-03-11-ablang2-rodriguez
site_sub_probs_df = pd.read_csv("_ignore/rodriguez_ablang2_ssp_df.csv.gz", index_col=0, dtype={'site':'object'})
results = plot_sites_observed_vs_expected(site_sub_probs_df, ax, numbering)

ax.text(
    0.02, 0.9,
    f'overlap={results["overlap"]:.3g}\nresidual={results["residual"]:.3g}',
    verticalalignment ='top',
    horizontalalignment ='left',
    transform = ax.transAxes,
    fontsize=15
)

ax.set_xlabel("sites (IMGT aligned)", fontsize=16)
ax.set_ylabel("number of substitutions", fontsize=18)
#ax.legend(loc='upper right', fontsize=12)
plt.setp(ax.get_xticklabels()[1::2], visible=False)
# ax.tick_params(axis="x", labelsize=12, labelrotation=90)

ax.set_title("AbLang2", fontsize=20)
plt.tight_layout()
outfname = f"{figures_dir}/sites-oe-rodriguez-all-ablang2.svg"
print(outfname,'created!')
plt.savefig(outfname)
plt.close()
