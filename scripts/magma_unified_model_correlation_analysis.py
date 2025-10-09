#!/usr/bin/env python3
"""
Unified MAGMA-seq Model Correlation Analysis

Creates lattice subplot visualization comparing experimental KD values with model predictions
for all antibody systems in the unified dataset (Kirby + Petersen).

Generates scatter plots showing model prediction vs -log10(KD) with correlation coefficients.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def load_scored_dataset():
    """Load the unified scored dataset."""
    scored_file = "data/whitehead/unified/magma_unified_scored.csv"
    
    if not os.path.exists(scored_file):
        print(f"❌ Dataset not found: {scored_file}")
        print("Please run: python3 scripts/magma_unified_pipeline.py")
        return None
        
    print(f"📂 Loading dataset: {scored_file}")
    df = pd.read_csv(scored_file)
    print(f"📊 Loaded {len(df)} sequences from {df['antibody'].nunique()} antibodies")
    
    # Add -log10(KD) for correlation analysis
    # KD values are in nM, so convert to proper -log10(KD): -log10(KD_nM * 1e-9) = -log10(KD_nM) + 9
    df['-log10_KD'] = -np.log10(df['KD']) + 9
    
    # ESM, AbLang, and ProGen scores are already negated in the scoring pipeline
    # (raw perplexity -> negated so higher score = better binding)
    # No additional negation needed here
    
    return df

def prepare_model_data(df):
    """Prepare data for model comparison analysis."""
    
    # Define models and their display names
    model_mapping = {
        'dasm_base': 'DASM',
        'esm': 'ESM-650M',
        'ablang': 'AbLang2',
        'progen': 'ProGen2'
    }
    
    # Check which models have valid data
    available_models = {}
    for model_col, display_name in model_mapping.items():
        if model_col in df.columns:
            valid_count = df[model_col].notna().sum()
            if valid_count > 0:
                available_models[model_col] = display_name
                print(f"✅ {display_name}: {valid_count}/{len(df)} sequences scored")
            else:
                print(f"⚠️  {display_name}: No valid scores found")
        else:
            print(f"❌ {display_name}: Column not found")
    
    return available_models

def create_correlation_plot(df, available_models, color_by_mutations=False):
    """Create lattice subplot correlation analysis.
    
    Args:
        df: DataFrame with scored data
        available_models: Dict of available models
        color_by_mutations: If True, color points by mutation count using turbo colormap
    """
    
    # Get unique antibodies sorted by dataset (Petersen first, then Kirby)
    # Exclude Ab_1-20_UCA due to small sample size (n=13)
    all_antibodies = df['antibody'].unique()
    petersen_antibodies = sorted([ab for ab in all_antibodies if df[df['antibody'] == ab]['dataset'].iloc[0] == 'Petersen'])
    kirby_antibodies = sorted([ab for ab in all_antibodies if df[df['antibody'] == ab]['dataset'].iloc[0] == 'Kirby' and ab != 'Ab_1-20_UCA'])
    antibodies = petersen_antibodies + kirby_antibodies
    # Order models alphabetically
    model_order = ['ablang', 'dasm_base', 'esm', 'progen']
    models = [m for m in model_order if m in available_models.keys()]
    
    color_suffix = "_mutations" if color_by_mutations else ""
    print(f"📊 Creating correlation plots{color_suffix} for {len(antibodies)} antibodies and {len(models)} models")
    
    # Set up the figure with proper spacing
    fig, axes = plt.subplots(len(models), len(antibodies), 
                            figsize=(4 * len(antibodies), 3 * len(models)), 
                            sharex='col')
    
    # Handle case where we have only one model or one antibody
    if len(models) == 1:
        axes = axes.reshape(1, -1)
    if len(antibodies) == 1:
        axes = axes.reshape(-1, 1)
    
    if color_by_mutations:
        # Get global mutation range for consistent colorbar
        global_mutation_range = df['mutations_from_reference'].dropna()
        vmin, vmax = global_mutation_range.min(), global_mutation_range.max()
        cmap = plt.cm.turbo
    else:
        # Color scheme - use different colors for Kirby vs Petersen
        color_map = {
            'Kirby': '#2E8B57',    # Sea green
            'Petersen': '#4682B4'  # Steel blue
        }
    
    for i, model in enumerate(models):
        for j, antibody in enumerate(antibodies):
            ax = axes[i, j]
            
            # Filter data for this antibody
            antibody_data = df[df['antibody'] == antibody].copy()
            
            # Remove rows with missing model scores or KD values
            valid_data = antibody_data.dropna(subset=[model, 'KD', '-log10_KD'])
            
            if len(valid_data) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{antibody} (n=0)', fontsize=10)
                continue
            
            if color_by_mutations:
                # Color by mutation count using turbo colormap
                mutation_counts = valid_data['mutations_from_reference']
                scatter = ax.scatter(valid_data['-log10_KD'], valid_data[model], 
                                   c=mutation_counts, cmap=cmap, vmin=vmin, vmax=vmax,
                                   alpha=0.7, s=25, edgecolors='none')
                
                # Add colorbar for the rightmost column
                if j == len(antibodies) - 1:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Mutations from reference', rotation=270, labelpad=15)
            else:
                # Determine dataset type for color
                dataset_type = valid_data['dataset'].iloc[0]
                color = color_map.get(dataset_type, '#666666')
                
                # Create scatter plot
                ax.scatter(valid_data['-log10_KD'], valid_data[model], 
                          color=color, alpha=0.6, s=25, edgecolors='none')
            
            # Calculate Pearson correlation only
            try:
                pearson_corr, _ = pearsonr(valid_data['-log10_KD'], valid_data[model])
                
                # Add correlation text (Pearson only, purple, no box, larger font)
                ax.text(0.05, 0.95, f'r = {pearson_corr:.3f}', transform=ax.transAxes,
                       color='purple', verticalalignment='top', fontsize=12, fontweight='bold')
                    
            except Exception as e:
                print(f"⚠️  Correlation calculation failed for {antibody} - {model}: {e}")
            
            # Set title for top row
            if i == 0:
                n_seqs = len(valid_data)
                dataset_info = f"({valid_data['dataset'].iloc[0]})" if not color_by_mutations else ""
                ax.set_title(f'{antibody} {dataset_info}\n(n={n_seqs})', fontsize=11, fontweight='bold')
            
            # Set x-axis label for bottom row
            if i == len(models) - 1:
                ax.set_xlabel(r'$-\log_{10}$(KD [nM])', fontsize=11)
            
            # Set y-axis label for left column
            if j == 0:
                model_display = available_models[model]
                if model in ['esm', 'ablang', 'progen']:
                    y_label = f'{model_display}\n(negative perplexity)'
                elif model in ['dasm_base']:
                    y_label = f'{model_display}\n(log selection score)'
                else:
                    y_label = f'{model_display}\nScore'
                ax.set_ylabel(y_label, fontsize=11)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set reasonable axis limits based on data
            if len(valid_data) > 1:
                x_margin = (valid_data['-log10_KD'].max() - valid_data['-log10_KD'].min()) * 0.1
                y_margin = (valid_data[model].max() - valid_data[model].min()) * 0.1
                
                ax.set_xlim(valid_data['-log10_KD'].min() - x_margin, 
                           valid_data['-log10_KD'].max() + x_margin)
                ax.set_ylim(valid_data[model].min() - y_margin,
                           valid_data[model].max() + y_margin)
    
    plt.tight_layout()
    
    return fig

def create_latex_table_and_csv(df, available_models):
    """Create transposed LaTeX table and CSV file with correlation results."""
    
    # Get unique antibodies (exclude Ab_1-20_UCA)
    all_antibodies = df['antibody'].unique()
    petersen_antibodies = sorted([ab for ab in all_antibodies if df[df['antibody'] == ab]['dataset'].iloc[0] == 'Petersen'])
    kirby_antibodies = sorted([ab for ab in all_antibodies if df[df['antibody'] == ab]['dataset'].iloc[0] == 'Kirby' and ab != 'Ab_1-20_UCA'])
    
    # Order models alphabetically
    model_order = ['ablang', 'dasm_base', 'esm', 'progen']
    models = [m for m in model_order if m in available_models.keys()]
    model_short_names = {
        'dasm_base': 'DASM',
        'esm': 'ESM',
        'ablang': 'AbLang2',
        'progen': 'ProGen2'
    }
    
    # Clean antibody names for display (remove _UCA suffix)
    def clean_antibody_name(name):
        return name.replace('_UCA', '')
    
    # Calculate correlations for all antibodies
    all_antibodies_data = {}
    csv_results = []
    
    # Process Petersen antibodies
    for antibody in petersen_antibodies:
        antibody_data = df[df['antibody'] == antibody].copy()
        valid_data_base = antibody_data.dropna(subset=['-log10_KD'])
        
        if len(valid_data_base) < 4:
            continue
            
        n_sequences = len(valid_data_base)
        correlations = {}
        
        for model in models:
            if model in antibody_data.columns:
                valid_data = antibody_data.dropna(subset=[model, '-log10_KD'])
                if len(valid_data) > 3:
                    pearson_corr, _ = pearsonr(valid_data['-log10_KD'], valid_data[model])
                    correlations[model] = pearson_corr
                else:
                    correlations[model] = None
            else:
                correlations[model] = None
        
        all_antibodies_data[antibody] = {
            'n_sequences': n_sequences,
            'correlations': correlations,
            'dataset': 'Petersen'
        }
        
        # CSV row
        csv_row = {'antibody': clean_antibody_name(antibody), 'n_sequences': n_sequences}
        for model in models:
            csv_row[model_short_names[model]] = correlations[model]
        csv_results.append(csv_row)
    
    # Process Kirby antibodies  
    for antibody in kirby_antibodies:
        antibody_data = df[df['antibody'] == antibody].copy()
        valid_data_base = antibody_data.dropna(subset=['-log10_KD'])
        
        if len(valid_data_base) < 4:
            continue
            
        n_sequences = len(valid_data_base)
        correlations = {}
        
        for model in models:
            if model in antibody_data.columns:
                valid_data = antibody_data.dropna(subset=[model, '-log10_KD'])
                if len(valid_data) > 3:
                    pearson_corr, _ = pearsonr(valid_data['-log10_KD'], valid_data[model])
                    correlations[model] = pearson_corr
                else:
                    correlations[model] = None
            else:
                correlations[model] = None
        
        all_antibodies_data[antibody] = {
            'n_sequences': n_sequences,
            'correlations': correlations,
            'dataset': 'Kirby'
        }
        
        # CSV row
        csv_row = {'antibody': clean_antibody_name(antibody), 'n_sequences': n_sequences}
        for model in models:
            csv_row[model_short_names[model]] = correlations[model]
        csv_results.append(csv_row)
    
    # Save CSV
    import pandas as pd
    results_df = pd.DataFrame(csv_results)
    csv_file = "data/whitehead/processed/magma_correlation_table.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"📊 CSV table saved: {csv_file}")
    
    # Create LaTeX table with multirow format
    latex_header = "\\begin{table}[ht]\n\\centering\n\\caption{Pearson correlations between model predictions and binding affinity}\n\\label{tab:magma-correlations}\n"
    latex_header += "\\begin{tabular}{ll|" + "c" * len(models) + "}\n\\toprule\n"
    
    # Header row with model names
    latex_header += "Source & Antibody"
    for model in models:
        latex_header += f" & {model_short_names[model]}"
    latex_header += " \\\\\n\\midrule\n"
    
    latex_rows = []
    
    # Add Petersen section with multirow
    if petersen_antibodies:
        petersen_antibodies_with_data = [ab for ab in petersen_antibodies if ab in all_antibodies_data]
        if petersen_antibodies_with_data:
            first_antibody = True
            for i, antibody in enumerate(petersen_antibodies_with_data):
                data = all_antibodies_data[antibody]
                correlations = data['correlations']
                
                # Find best model for this antibody
                valid_correlations = {k: v for k, v in correlations.items() if v is not None}
                best_model = max(valid_correlations.keys(), key=lambda k: valid_correlations[k]) if valid_correlations else None
                
                if first_antibody:
                    latex_row = f"\\multirow{{{len(petersen_antibodies_with_data)}}}{{*}}{{Petersen~\\cite{{Petersen2024-ud}}}} & {clean_antibody_name(antibody)}"
                    first_antibody = False
                else:
                    latex_row = f"& {clean_antibody_name(antibody)}"
                    
                for model in models:
                    if correlations[model] is not None:
                        corr_str = f"{correlations[model]:.3f}"
                        if model == best_model:
                            latex_row += f" & \\textbf{{{corr_str}}}"
                        else:
                            latex_row += f" & {corr_str}"
                    else:
                        latex_row += " & --"
                latex_row += " \\\\"
                latex_rows.append(latex_row)
    
    # Add separator
    if petersen_antibodies and kirby_antibodies:
        latex_rows.append("\\midrule")
    
    # Add Kirby section with multirow
    if kirby_antibodies:
        kirby_antibodies_with_data = [ab for ab in kirby_antibodies if ab in all_antibodies_data]
        if kirby_antibodies_with_data:
            first_antibody = True
            for i, antibody in enumerate(kirby_antibodies_with_data):
                data = all_antibodies_data[antibody]
                correlations = data['correlations']
                
                # Find best model for this antibody
                valid_correlations = {k: v for k, v in correlations.items() if v is not None}
                best_model = max(valid_correlations.keys(), key=lambda k: valid_correlations[k]) if valid_correlations else None
                
                if first_antibody:
                    latex_row = f"\\multirow{{{len(kirby_antibodies_with_data)}}}{{*}}{{Kirby~\\cite{{Kirby2025-rq}}}} & {clean_antibody_name(antibody)}"
                    first_antibody = False
                else:
                    latex_row = f"& {clean_antibody_name(antibody)}"
                    
                for model in models:
                    if correlations[model] is not None:
                        corr_str = f"{correlations[model]:.3f}"
                        if model == best_model:
                            latex_row += f" & \\textbf{{{corr_str}}}"
                        else:
                            latex_row += f" & {corr_str}"
                    else:
                        latex_row += " & --"
                latex_row += " \\\\"
                latex_rows.append(latex_row)
    
    latex_footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"
    
    latex_table = latex_header + "\n".join(latex_rows) + "\n" + latex_footer
    
    # Save LaTeX table
    latex_file = "data/whitehead/processed/magma_correlation_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"📄 LaTeX table saved: {latex_file}")
    
    # Print LaTeX table to console
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print(latex_table)
    
    return results_df, latex_table

def print_summary_statistics(df, available_models):
    """Print summary statistics for the analysis."""
    print("\n" + "="*80)
    print("UNIFIED MAGMA-SEQ MODEL CORRELATION SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"Total sequences: {len(df)}")
    print(f"Antibody systems: {df['antibody'].nunique()}")
    print(f"Datasets: {', '.join(df['dataset'].unique())}")
    print()
    
    # Per-antibody statistics
    print("Per-antibody sequence counts:")
    antibody_counts = df['antibody'].value_counts().sort_index()
    for antibody, count in antibody_counts.items():
        dataset = df[df['antibody'] == antibody]['dataset'].iloc[0]
        print(f"  {antibody} ({dataset}): {count} sequences")
    print()
    
    # Model coverage statistics
    print("Model scoring coverage:")
    for model_col, display_name in available_models.items():
        valid_count = df[model_col].notna().sum()
        coverage = valid_count / len(df) * 100
        print(f"  {display_name}: {valid_count}/{len(df)} ({coverage:.1f}%)")
    print()
    
    # Overall correlations (Pearson only)
    print("Overall correlations with -log10(KD) (Pearson r):")
    for model_col, display_name in available_models.items():
        valid_data = df.dropna(subset=[model_col, '-log10_KD'])
        if len(valid_data) > 1:
            pearson_corr, _ = pearsonr(valid_data['-log10_KD'], valid_data[model_col])
            print(f"  {display_name}: r = {pearson_corr:.3f}")
        else:
            print(f"  {display_name}: insufficient data")

def main():
    print("="*80)
    print("UNIFIED MAGMA-SEQ MODEL CORRELATION ANALYSIS")
    print("="*80)
    
    # Load scored dataset
    df = load_scored_dataset()
    if df is None:
        return
    
    # Prepare model data
    available_models = prepare_model_data(df)
    if not available_models:
        print("❌ No models with valid data found")
        return
    
    # Create standard correlation plot (colored by dataset)
    fig1 = create_correlation_plot(df, available_models, color_by_mutations=False)
    
    # Save standard plot
    output_file1 = "data/whitehead/processed/magma_unified_model_correlations.svg"
    os.makedirs(os.path.dirname(output_file1), exist_ok=True)
    fig1.savefig(output_file1, bbox_inches='tight', dpi=300)
    print(f"📊 Standard correlation plot saved: {output_file1}")
    
    # Create mutation-colored correlation plot
    fig2 = create_correlation_plot(df, available_models, color_by_mutations=True)
    
    # Save mutation-colored plot
    output_file2 = "data/whitehead/processed/magma_unified_model_correlations_mutations.svg"
    fig2.savefig(output_file2, bbox_inches='tight', dpi=300)
    print(f"📊 Mutation-colored correlation plot saved: {output_file2}")
    
    # Create LaTeX table and CSV
    results_df, latex_table = create_latex_table_and_csv(df, available_models)
    
    # Print summary statistics
    print_summary_statistics(df, available_models)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"Standard visualization: {output_file1}")
    print(f"Mutation-colored visualization: {output_file2}")
    print(f"Open standard plot: open {output_file1}")
    print(f"Open mutation plot: open {output_file2}")
    
    return output_file1, output_file2

if __name__ == "__main__":
    main()