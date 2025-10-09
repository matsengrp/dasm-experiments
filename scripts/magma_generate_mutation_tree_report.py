#!/usr/bin/env python3
"""
Generate MAGMA-seq mutation tree visualization report directly as HTML.
Bypasses notebook execution issues by creating charts programmatically.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import altair as alt
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dnsmex.magma_tree_analysis import (
    load_uca_partition_data,
    identify_parent_child_pairs,
    create_mutation_plot,
    analyze_tree_metrics
)

# Configure Altair to embed data directly in JSON
alt.data_transformers.enable('default', max_rows=None)

def create_unified_charts():
    """Create charts for all available antibody systems from both datasets."""
    
    # Change to project root
    os.chdir(project_root)
    print(f"Working from: {os.getcwd()}")
    
    # Find partition files from both datasets
    import glob
    kirby_partitions = glob.glob('data/whitehead/kirby/processed/*_partition.csv')
    petersen_partitions = glob.glob('data/whitehead/petersen/processed/*_partition.csv')
    partition_files = kirby_partitions + petersen_partitions
    
    print(f"Found {len(partition_files)} antibody systems:")
    for f in partition_files:
        print(f"  - {f}")
    
    charts = {}
    summary_data = []
    
    for partition_file in partition_files:
        # Extract system name and dataset
        system_name = Path(partition_file).stem.replace('_partition', '')
        dataset = 'Kirby' if 'kirby' in partition_file else 'Petersen'
        
        print(f"\n📊 Processing {system_name} ({dataset})...")
        
        try:
            # Load partition data directly
            tree_data = pd.read_csv(partition_file)
            
            # Clean up synthetic UCA entries but keep actual experimental observations
            if dataset == 'Petersen':
                print(f"🧹 Cleaning Petersen data for {system_name}...")
                
                # Remove synthetic UCA sequences (the 1500 nM ones we added)
                synthetic_uca_mask = (tree_data["mutations_from_uca"] == 0) & (tree_data["KD"] == 1500.0)
                tree_data = tree_data[~synthetic_uca_mask]
                print(f"🗑️  Removed {synthetic_uca_mask.sum()} synthetic UCA entries")
                print(f"✅ Cleaned data shape: {tree_data.shape}")
            elif dataset == 'Kirby':
                print(f"🧹 Cleaning Kirby data for {system_name}...")
                
                # Remove synthetic UCA sequences (1500 nM) but keep actual experimental UCAs
                synthetic_uca_mask = (tree_data["mutations_from_uca"] == 0) & (tree_data["KD"] == 1500.0)
                tree_data = tree_data[~synthetic_uca_mask] 
                print(f"🗑️  Removed {synthetic_uca_mask.sum()} synthetic UCA entries")
                
                # Check if we have any actual experimental UCAs remaining
                actual_uca_count = ((tree_data["mutations_from_uca"] == 0) & (tree_data["KD"] != 1500.0)).sum()
                print(f"📊 Actual experimental UCAs remaining: {actual_uca_count}")
                print(f"✅ Cleaned data shape: {tree_data.shape}")
            
            # Use existing analysis functions (adapting for new format)
            edges = identify_parent_child_pairs(tree_data, system_name)
            metrics = analyze_tree_metrics(tree_data, edges)
            
            # Store metrics with dataset info
            summary_data.append({
                'antibody_system': system_name,
                'dataset': dataset,
                'n_sequences': metrics['n_sequences'], 
                'n_edges': metrics['n_edges'],
                'max_mutations': metrics['max_mutations'],
                'mean_kd': metrics['mean_kd']
            })
            
            # Create chart with dataset info in title
            chart_title = f'{system_name} ({dataset}) Mutation Tree'
            chart = create_mutation_plot(tree_data, edges, chart_title, dataset_type=dataset)
            charts[f"{system_name}_{dataset}"] = chart
            
            print(f"✅ Created chart for {system_name}: {metrics['n_sequences']} sequences, {metrics['n_edges']} edges")
            
        except Exception as e:
            print(f"❌ Error processing {system_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return charts, pd.DataFrame(summary_data) if summary_data else None

def create_comparison_chart(summary_df):
    """Create comparison chart across antibody systems."""
    if summary_df is None or len(summary_df) == 0:
        return None
        
    return alt.Chart(summary_df).mark_bar().encode(
        x=alt.X('antibody_system:N', 
                title='Antibody System',
                axis=alt.Axis(titleFontSize=12, labelFontSize=10, labelAngle=-45)),
        y=alt.Y('improvement_rate:Q', 
                title='Improvement Rate (%)', 
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(titleFontSize=12, labelFontSize=10, format='%')),
        color=alt.Color('dataset:N', 
                       title='Dataset',
                       legend=alt.Legend(orient='top')),
        tooltip=[
            alt.Tooltip('antibody_system:N', title='Antibody System'),
            alt.Tooltip('dataset:N', title='Dataset'),
            alt.Tooltip('n_sequences:Q', title='Number of Sequences'),
            alt.Tooltip('n_edges:Q', title='Parent-Child Edges'),
            alt.Tooltip('improvement_rate:Q', title='Improvement Rate', format='.1%')
        ]
    ).properties(
        width=500,
        height=300,
        title=alt.TitleParams(text='Improvement Rate by Antibody System', fontSize=14, fontWeight='bold')
    )

def generate_html_report(charts, summary_df):
    """Generate complete HTML report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MAGMA-seq Unified Mutation Tree Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 40px; }}
        .chart {{ margin: 20px 0; }}
        .metrics {{ 
            background: #f5f5f5; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0; 
        }}
        .summary-table {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
        .dataset-kirby {{ color: #2E8B57; }}
        .dataset-petersen {{ color: #4682B4; }}
    </style>
</head>
<body>
    <h1>MAGMA-seq Unified Mutation Tree Visualization</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    
    <h2>Overview</h2>
    <p>Interactive visualization of parent-child mutation relationships in antibody affinity maturation using Dr. Whitehead's authoritative reference sequences:</p>
    <ul>
        <li><strong>Datasets</strong>: <span class="dataset-kirby">Kirby et al. 2025</span> and <span class="dataset-petersen">Petersen et al. 2024</span></li>
        <li><strong>X-axis</strong>: Number of mutations from reference sequence (0 = reference position)</li>
        <li><strong>Y-axis</strong>: Binding affinity (KD in nM, log scale)</li>
        <li><strong>Points</strong>: Individual experimental sequences</li>
        <li><strong>Interactivity</strong>: Hover for sequence details</li>
    </ul>
"""

    # Add individual UCA group charts
    for group, chart in charts.items():
        html_content += f"""
    <h2>{group} Mutation Tree</h2>
    <div class="chart" id="chart_{group}"></div>
    <script type="text/javascript">
        vegaEmbed('#chart_{group}', {chart.to_json()});
    </script>
"""

    # Skip comparative analysis section

    # Add summary table
    if summary_df is not None:
        summary_html = summary_df.round(3).to_html(index=False, classes="summary-table")
        html_content += f"""
    <h2>Summary Statistics</h2>
    {summary_html}
"""

    html_content += """
    <h2>Export Options</h2>
    <p>Charts can be exported by right-clicking and selecting "Save as PNG" or "View Source" for SVG.</p>
    
</body>
</html>
"""
    
    return html_content

def main():
    """Main execution function."""
    print("🚀 Generating unified MAGMA-seq mutation tree visualization report...")
    
    # Create charts for all antibody systems
    charts, summary_df = create_unified_charts()
    
    if not charts:
        print("❌ No charts created - check data files")
        return
    
    # Generate HTML report
    html_content = generate_html_report(charts, summary_df)
    
    # Save report with new name
    # Create processed directory if it doesn't exist
    os.makedirs("data/whitehead/processed", exist_ok=True)
    output_file = "data/whitehead/processed/magma_mutation_trees.html"
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Report generated: {output_file}")
    print(f"📊 Charts created: {len(charts)}")
    if summary_df is not None:
        print(f"📈 Antibody systems analyzed: {len(summary_df)}")
    
    return output_file

if __name__ == "__main__":
    main()