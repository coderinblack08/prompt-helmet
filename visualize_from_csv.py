import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.interpolate import interp1d

def create_output_directory():
    """Create a directory for the output"""
    os.makedirs("ablation_analysis", exist_ok=True)
    return "ablation_analysis"

def visualize_from_csv(csv_path, output_dir):
    """Create visualizations from a pre-existing CSV file"""
    print(f"Reading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    head_columns = [col for col in df.columns if col.startswith('Head_')]
    impact_matrix = df[head_columns].values
    
    num_layers = impact_matrix.shape[0]
    num_heads = impact_matrix.shape[1]
    
    split_idx = num_layers // 2
    first_half_layers = impact_matrix[:split_idx, :]
    second_half_layers = impact_matrix[split_idx:, :]
    first_half_layer_indices = df['Layer'].values[:split_idx]
    second_half_layer_indices = df['Layer'].values[split_idx:]
    
    print(f"Found data for {num_layers} layers and {num_heads} heads")
    print(f"Splitting layers into two groups: 0-{split_idx-1} and {split_idx}-{num_layers-1}")
    
    vmin = np.min(impact_matrix)
    vmax = np.max(impact_matrix)
    
    print(f"Raw data range: min={vmin:.4f}, max={vmax:.4f}")
    
    abs_min = abs(vmin)
    vmax = abs_min
    
    print(f"Adjusted data range: min={vmin:.4f}, max={vmax:.4f}")
    
    width_per_head = 2.5
    height_per_layer = 1.6
    fig_width = max(30, num_heads * width_per_head * 2)
    fig_height = max(22, (num_layers // 2) * height_per_layer)
    
    fig, (ax_heatmap1, ax_heatmap2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), 
                                                  gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.0})
    
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    
    fontsize = max(18, min(24, 800 / (num_heads * (num_layers // 2))))
    
    def create_heatmap(ax, data, ylabels, show_cbar=False):
        h = sns.heatmap(
            data, 
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            annot=True, 
            fmt=".3f",
            linewidths=1.5,
            annot_kws={"size": fontsize, "weight": "bold"},
            square=True,
            cbar=show_cbar,
            cbar_kws={'shrink': 0.7, 'location': 'right'} if show_cbar else None,
            xticklabels=[col.replace('Head_', '') for col in head_columns],
            yticklabels=ylabels
        )
        
        ax.set_xlabel("Head Index", fontsize=26, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        for i in range(1, num_heads):
            ax.axvline(i, color='white', linewidth=1.5)
        
        for i in range(1, data.shape[0]):
            ax.axhline(i, color='white', linewidth=2.5)
        
        ax.invert_yaxis()
        
        return h
    
    h1 = create_heatmap(ax_heatmap1, first_half_layers, first_half_layer_indices, show_cbar=False)
    h2 = create_heatmap(ax_heatmap2, second_half_layers, second_half_layer_indices, show_cbar=False)
    
    ax_heatmap1.set_ylabel("Layer Index", fontsize=26, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    cax = plt.axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(h2.collections[0], cax=cax)
    
    formula_label = "Distraction Score = sum(A_instr(corrupted)) - sum(A_instr(clean))"
    
    cbar.ax.set_ylabel(formula_label, fontsize=22, fontweight='bold', labelpad=20)
    cbar.ax.tick_params(labelsize=20)
    
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    
    output_path = f"{output_dir}/head_ablation_compact_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    return output_path

def main():
    """Main function to read the CSV and generate visualizations"""
    output_dir = create_output_directory()
    
    csv_path = "distraction_analysis/distraction_impacts.csv"
    
    visualize_from_csv(csv_path, output_dir)
    
    print("Visualization completed!")

if __name__ == "__main__":
    main() 