import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def create_output_directory():
    """Create a directory for the output"""
    os.makedirs("attention_analysis", exist_ok=True)
    return "attention_analysis"

def visualize_differences(json_path, output_dir):
    """Create visualizations from the differences.json file"""
    print(f"Reading data from: {json_path}")
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract attention differences, which is a 3D array: [example, layer, head]
    differences = np.array(data["differences"])
    
    # Get dimensions
    num_examples = differences.shape[0]
    num_layers = differences.shape[1]
    num_heads = differences.shape[2]
    
    print(f"Found data for {num_examples} examples, {num_layers} layers and {num_heads} heads")
    
    # For each example, create a heatmap
    for example_idx in range(num_examples):
        impact_matrix = differences[example_idx]
        
        # Get the actual min and max values for better scaling
        vmin = np.min(impact_matrix)
        vmax = np.max(impact_matrix)
        
        print(f"Example {example_idx} data range: min={vmin:.4f}, max={vmax:.4f}")
        
        # Calculate figure dimensions based on number of heads and layers
        width_per_head = 1.5
        height_per_layer = 1.0
        fig_width = max(15, num_heads * width_per_head)
        fig_height = max(10, num_layers * height_per_layer)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create a diverging colormap: red for negative impacts, blue for positive impacts
        cmap = sns.diverging_palette(10, 220, as_cmap=True)
        
        # Fontsize for annotations
        fontsize = max(8, min(12, 400 / (num_heads * num_layers)))
        
        # Create the heatmap
        h = sns.heatmap(
            impact_matrix, 
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            annot=True, 
            fmt=".3f",
            linewidths=0.5,
            annot_kws={"size": fontsize},
            cbar=True,
            cbar_kws={'shrink': 0.7, 'label': 'Attention Difference (Corrupted - Clean)'},
            xticklabels=range(num_heads),
            yticklabels=range(num_layers)
        )
        
        # Add labels
        ax.set_xlabel("Head Index", fontsize=12, fontweight='bold')
        ax.set_ylabel("Layer Index", fontsize=12, fontweight='bold')
        ax.set_title(f"Attention Map Differences (Example {example_idx})", fontsize=14, fontweight='bold')
        
        # Add vertical lines to group heads within each layer
        for i in range(1, num_heads):
            ax.axvline(i, color='white', linewidth=0.5)
        
        # Add horizontal lines between layers
        for i in range(1, num_layers):
            ax.axhline(i, color='white', linewidth=0.5)
        
        # Save the figure
        output_path = f"{output_dir}/attention_difference_example_{example_idx}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
    
    # Now create layer-wise analysis (average across examples)
    avg_differences = np.mean(differences, axis=0)
    
    # Get the min and max values
    vmin = np.min(avg_differences)
    vmax = np.max(avg_differences)
    
    print(f"Average differences data range: min={vmin:.4f}, max={vmax:.4f}")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create the heatmap for average differences
    h = sns.heatmap(
        avg_differences, 
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        annot=True, 
        fmt=".3f",
        linewidths=0.5,
        annot_kws={"size": fontsize},
        cbar=True,
        cbar_kws={'shrink': 0.7, 'label': 'Average Attention Difference'},
        xticklabels=range(num_heads),
        yticklabels=range(num_layers)
    )
    
    # Add labels
    ax.set_xlabel("Head Index", fontsize=12, fontweight='bold')
    ax.set_ylabel("Layer Index", fontsize=12, fontweight='bold')
    ax.set_title("Average Attention Map Differences", fontsize=14, fontweight='bold')
    
    # Add vertical and horizontal lines
    for i in range(1, num_heads):
        ax.axvline(i, color='white', linewidth=0.5)
    for i in range(1, num_layers):
        ax.axhline(i, color='white', linewidth=0.5)
    
    # Save the figure
    output_path = f"{output_dir}/attention_difference_average.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Average visualization saved to: {output_path}")
    
    # Also create a comparative visualization of clean vs corrupted attention
    clean_scores = np.array(data["clean_scores"])
    corrupted_scores = np.array(data["corrupted_scores"])
    
    # Create a 2x1 subplot for clean vs corrupted
    for example_idx in range(num_examples):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width * 2, fig_height), 
                                       gridspec_kw={'width_ratios': [1, 1]})
        
        # Find common min and max for both heatmaps
        vmin = min(np.min(clean_scores[example_idx]), np.min(corrupted_scores[example_idx]))
        vmax = max(np.max(clean_scores[example_idx]), np.max(corrupted_scores[example_idx]))
        
        # Create heatmap for clean attention
        sns.heatmap(
            clean_scores[example_idx], 
            ax=ax1,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            annot=True, 
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": fontsize},
            cbar=True,
            cbar_kws={'shrink': 0.7, 'label': 'Attention Score'},
            xticklabels=range(num_heads),
            yticklabels=range(num_layers)
        )
        
        # Create heatmap for corrupted attention
        sns.heatmap(
            corrupted_scores[example_idx], 
            ax=ax2,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            annot=True, 
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": fontsize},
            cbar=True,
            cbar_kws={'shrink': 0.7, 'label': 'Attention Score'},
            xticklabels=range(num_heads),
            yticklabels=range(num_layers)
        )
        
        # Add titles and labels
        ax1.set_title("Clean Attention", fontsize=14, fontweight='bold')
        ax2.set_title("Corrupted Attention", fontsize=14, fontweight='bold')
        
        ax1.set_xlabel("Head Index", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Layer Index", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Head Index", fontsize=12, fontweight='bold')
        
        # Save the figure
        output_path = f"{output_dir}/clean_vs_corrupted_example_{example_idx}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison visualization saved to: {output_path}")
    
    return output_dir

def main():
    """Main function to read the JSON and generate visualizations"""
    # Create output directory
    output_dir = create_output_directory()
    
    # Path to the JSON file
    json_path = "differences.json"
    
    # Generate the visualizations
    visualize_differences(json_path, output_dir)
    
    print("Visualizations completed!")

if __name__ == "__main__":
    main() 