import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_output_directory():
    """Create a directory for all outputs"""
    os.makedirs("ablation_analysis", exist_ok=True)
    return "ablation_analysis"

def update_ablation_visualization(csv_path, output_dir):
    """
    Create an improved visualization from the CSV data with labeled color scale
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Extract the data for visualization
    num_layers = len(df)
    num_heads = len(df.columns) - 2  # Excluding Layer and Full_Layer_Impact
    
    # Create head impact matrix
    impact_matrix = df.iloc[:, 1:num_heads+1].values
    
    # Get layer impact values
    layer_total_impacts = df["Full_Layer_Impact"].values
    
    # Create the figure with extra space on the right for a different visualization
    fig, (ax_heatmap, ax_right) = plt.subplots(1, 2, figsize=(16, 8),
                                            gridspec_kw={'width_ratios': [3, 1]})
    
    # Create a heatmap showing the impacts with more potent color scale
    # and blue for positive impacts (inhibit extraction)
    # Using standard colors with more true blue instead of cyan
    cmap = sns.diverging_palette(230, 15, s=85, l=40, as_cmap=True)  # Darker red/true blue
    
    # Create the heatmap
    heatmap = sns.heatmap(
        impact_matrix, 
        ax=ax_heatmap,
        cmap=cmap,
        vmin=-0.03,  # More potent color scale
        vmax=0.03,
        center=0,
        annot=True, 
        fmt=".3f",
        linewidths=0.5,
        square=True,
        cbar_kws={
            "shrink": 0.8, 
            "label": r"$\Delta P(\text{secret token}) = P_{\text{patched}} - P_{\text{corrupted}}$"  # LaTeX formula for the scale
        }
    )
    
    ax_heatmap.set_xlabel("Head Index")
    ax_heatmap.set_ylabel("Layer Index")
    
    # Add vertical lines to group heads within each layer
    for i in range(1, num_heads):
        ax_heatmap.axvline(i, color='white', linewidth=0.5)
    
    # Add horizontal lines between layers
    for i in range(1, num_layers):
        ax_heatmap.axhline(i, color='white', linewidth=1.5)
    
    # Alternative visualization on the right: Dot plot with size and color encoding
    # Clear the right axis for our custom visualization
    ax_right.clear()
    
    # Get a consistent color palette
    palette = sns.diverging_palette(230, 15, s=85, l=40, as_cmap=False, n=256)
    
    # Set up the plot
    ax_right.axvline(0, color='gray', linestyle='-', alpha=0.3, zorder=0)
    ax_right.set_xlabel("Total Layer Impact")
    ax_right.set_title("Layer Impact Magnitude", fontsize=10)
    ax_right.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Calculate color and size mappings
    norm = plt.Normalize(-max(abs(min(layer_total_impacts)), abs(max(layer_total_impacts))), 
                         max(abs(min(layer_total_impacts)), abs(max(layer_total_impacts))))
    
    # Display layer impacts as a dot plot with gradient
    y_pos = np.arange(num_layers)
    
    for i, impact in enumerate(layer_total_impacts):
        # Calculate color based on impact sign and magnitude
        color_idx = int((impact / max(abs(min(layer_total_impacts)), abs(max(layer_total_impacts))) + 1) * 128)
        color_idx = max(0, min(color_idx, 255))  # Ensure within bounds
        
        # Size proportional to absolute impact
        size = 100 + 900 * (abs(impact) / max(abs(min(layer_total_impacts)), abs(max(layer_total_impacts))))
        
        # Draw the dot
        ax_right.scatter(impact, i, s=size, color=palette[color_idx], 
                      edgecolor='black', linewidth=1, alpha=0.8, zorder=2)
        
        # Add impact value as text
        if abs(impact) > 0.5:
            weight = 'bold'
            fontsize = 10
        else:
            weight = 'normal'
            fontsize = 9
            
        ax_right.text(
            impact + (0.001 if impact >= 0 else -0.001), 
            i,
            f"{impact:.3f}", 
            ha='left' if impact >= 0 else 'right',
            va='center',
            fontweight=weight,
            fontsize=fontsize,
            zorder=3
        )
    
    # Set the y-ticks to match the layers
    ax_right.set_yticks(np.arange(num_layers))
    ax_right.set_yticklabels([f"Layer {i}" for i in range(num_layers)])
    
    # Set reasonable x-axis limits
    max_abs_impact = max(abs(min(layer_total_impacts)), abs(max(layer_total_impacts)))
    ax_right.set_xlim(-max_abs_impact * 1.2, max_abs_impact * 1.2)
    
    # Match y-axis limits with heatmap
    ax_right.set_ylim([num_layers - 0.5, -0.5])
    
    # No title as requested
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multilayer_head_ablation_impacts_labeled.png", dpi=200)
    plt.close()
    
    print(f"Updated visualization saved to {output_dir}/multilayer_head_ablation_impacts_labeled.png")

if __name__ == "__main__":
    output_dir = create_output_directory()
    csv_path = "ablation_analysis/ablation_impacts.csv"
    update_ablation_visualization(csv_path, output_dir) 