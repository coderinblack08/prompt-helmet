import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import torch
import math
from typing import List, Tuple, Dict, Optional, Union
import argparse
import json
import matplotlib.patches as patches

def create_output_directory():
    """Create a directory for the output"""
    os.makedirs("distraction_analysis", exist_ok=True)
    return "distraction_analysis"

def compute_distraction_effect(
    clean_attn_path: str,
    corrupted_attn_path: str,
    output_csv_path: str,
    num_layers: int = None, 
    num_heads: int = None
) -> np.ndarray:
    """
    Compute the distraction effect scores by comparing clean and corrupted attention patterns.
    
    Args:
        clean_attn_path: Path to the JSON file containing clean attention patterns
        corrupted_attn_path: Path to the JSON file containing corrupted attention patterns
        output_csv_path: Path to save the calculated distraction effect scores
        num_layers: Number of layers to analyze (automatically determined if None)
        num_heads: Number of heads per layer (automatically determined if None)
        
    Returns:
        Numpy array containing the distraction effect scores
    """
    print(f"Computing distraction effect from attention patterns...")
    print(f"  Clean attention data: {clean_attn_path}")
    print(f"  Corrupted attention data: {corrupted_attn_path}")
    
    # Load the clean and corrupted attention patterns
    try:
        with open(clean_attn_path, 'r') as f:
            clean_data = json.load(f)
        
        with open(corrupted_attn_path, 'r') as f:
            corrupted_data = json.load(f)
        
        # Extract attention patterns
        # The expected format is a nested dictionary:
        # { "layer_0": { "head_0": { "attention": [...] }, "head_1": {...} }, "layer_1": {...} }
        
        # First, determine dimensions if not provided
        if num_layers is None or num_heads is None:
            # Count number of layers and heads from the data
            num_layers = len(clean_data)
            # Assume all layers have the same number of heads
            num_heads = len(clean_data[list(clean_data.keys())[0]])
            
        print(f"Analyzing {num_layers} layers with {num_heads} heads each")
        
        # Initialize the matrix to store distraction effect scores
        distraction_matrix = np.zeros((num_layers, num_heads))
        
        # Process each layer and head
        for layer_idx in range(num_layers):
            layer_key = f"layer_{layer_idx}"
            
            if layer_key not in clean_data or layer_key not in corrupted_data:
                print(f"Warning: Layer {layer_idx} data not found in one or both attention files")
                continue
                
            clean_layer = clean_data[layer_key]
            corrupted_layer = corrupted_data[layer_key]
            
            for head_idx in range(num_heads):
                head_key = f"head_{head_idx}"
                
                if head_key not in clean_layer or head_key not in corrupted_layer:
                    print(f"Warning: Head {head_idx} in layer {layer_idx} not found in one or both attention files")
                    continue
                
                # Get attention patterns to instruction tokens
                # In actual implementation, you would extract the specific attention pattern
                # to the instruction tokens. For this simplified version, we'll just use
                # an aggregated attention score.
                
                clean_instr_attn = clean_layer[head_key].get("instruction_attention", 0)
                corrupted_instr_attn = corrupted_layer[head_key].get("instruction_attention", 0)
                
                # Calculate distraction effect: negative value means attention shifted away from instruction
                distraction_effect = corrupted_instr_attn - clean_instr_attn
                distraction_matrix[layer_idx, head_idx] = distraction_effect
        
        # Create a DataFrame to save the results
        layer_indices = list(range(num_layers))
        head_columns = [f"Head_{i}" for i in range(num_heads)]
        
        df = pd.DataFrame(distraction_matrix, columns=head_columns)
        df.insert(0, "Layer", layer_indices)
        
        # Add a column for the overall layer impact (mean across heads)
        df["Full_Layer_Impact"] = df[head_columns].mean(axis=1)
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Distraction effect scores saved to: {output_csv_path}")
        
        return distraction_matrix
        
    except Exception as e:
        print(f"Error computing distraction effect: {str(e)}")
        # Return an empty matrix with the specified dimensions or default 24x12
        if num_layers is None:
            num_layers = 24
        if num_heads is None:
            num_heads = 12
        return np.zeros((num_layers, num_heads))

def analyze_distraction_from_raw_data(
    clean_attn_path: str,
    corrupted_attn_path: str,
    output_dir: str,
    num_layers: int = None,
    num_heads: int = None
) -> str:
    """
    Analyze distraction effect from raw attention data and visualize the results.
    
    Args:
        clean_attn_path: Path to the clean attention data
        corrupted_attn_path: Path to the corrupted attention data
        output_dir: Directory to save outputs
        num_layers: Number of layers to analyze
        num_heads: Number of heads per layer
        
    Returns:
        Path to the generated visualization
    """
    # Compute distraction effect scores
    output_csv_path = os.path.join(output_dir, "distraction_impacts.csv")
    distraction_matrix = compute_distraction_effect(
        clean_attn_path, 
        corrupted_attn_path, 
        output_csv_path,
        num_layers,
        num_heads
    )
    
    # Visualize the results
    return visualize_distraction_from_csv(output_csv_path, output_dir)

def visualize_distraction_from_csv(csv_path, output_dir):
    """Create visualizations from a CSV file containing distraction effect data"""
    print(f"Reading distraction effect data from: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get head columns (all columns except Layer and any "Full_" columns)
    head_columns = [col for col in df.columns if col.startswith('Head_')]
    
    # Extract layer indices
    layer_indices = df['Layer'].values
    
    # Create distraction effect matrix for heatmap
    distraction_matrix = df[head_columns].values
    
    # Get dimensions
    num_layers = distraction_matrix.shape[0]
    num_heads = distraction_matrix.shape[1]
    
    print(f"Found data for {num_layers} layers and {num_heads} heads")
    
    # Calculate the optimal number of panels for the visualization
    # For 24 or fewer layers, use 2 panels
    # For more than 24 layers, use 3 panels
    num_panels = 2 if num_layers <= 28 else 3
    layers_per_panel = math.ceil(num_layers / num_panels)
    
    print(f"Using {num_panels} panels with approximately {layers_per_panel} layers per panel")
    
    # Split layers into panels
    panel_data = []
    panel_indices = []
    
    for i in range(num_panels):
        start_idx = i * layers_per_panel
        end_idx = min((i + 1) * layers_per_panel, num_layers)
        
        if start_idx < num_layers:  # Only add if we have layers for this panel
            panel_data.append(distraction_matrix[start_idx:end_idx, :])
            panel_indices.append(layer_indices[start_idx:end_idx])
            print(f"Panel {i+1}: Layers {start_idx}-{end_idx-1}")
    
    # Get the actual min and max values for better scaling
    vmin = np.min(distraction_matrix)
    vmax = np.max(distraction_matrix)
    
    # Ensure we have both positive and negative values in the scale
    # If all values are negative, create a balanced scale
    if vmax <= 0:
        vmax = abs(vmin)  # Make max match the magnitude of min
    # If all values are positive, create a balanced scale
    if vmin >= 0:
        vmin = -vmax  # Make min match the negative magnitude of max
    # Otherwise, use max absolute value for balanced scale
    abs_max = max(abs(vmin), abs(vmax))
    vmin = -abs_max
    vmax = abs_max
    
    print(f"Raw data range: min={np.min(distraction_matrix):.4f}, max={np.max(distraction_matrix):.4f}")
    print(f"Adjusted scale: min={vmin:.4f}, max={vmax:.4f}")
    
    # For distraction effect, we want to highlight negative values (indicating distraction)
    # and positive values (indicating increased attention)
    # We'll use a diverging colormap centered at 0
    
    # Calculate figure dimensions based on number of heads and layers
    width_per_head = 2.5  # Space for each head
    height_per_layer = 1.6  # Space for each layer
    fig_width = max(30, num_heads * width_per_head * num_panels)  # Width based on panels
    fig_height = max(22, layers_per_panel * height_per_layer)  # Height based on layers per panel
    
    # Create the figure with panels and space for colorbar
    fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, fig_height), 
                            gridspec_kw={'width_ratios': [1] * num_panels, 'wspace': 0.0})
    
    # Handle the case where there's only one panel (axes would be a single object, not an array)
    if num_panels == 1:
        axes = [axes]
    
    # Create a diverging colormap: red for negative impacts (distraction), blue for positive impacts
    cmap = sns.diverging_palette(10, 220, as_cmap=True)  # Red for negative, blue for positive
    
    # Font size for the annotations
    fontsize = max(18, min(24, 800 / (num_heads * layers_per_panel)))
    
    # Identify the top 20 most distracting heads (most negative values)
    flattened_data = distraction_matrix.flatten()
    sorted_indices = np.argsort(flattened_data)
    top_k = min(20, len(sorted_indices))
    top_distracting_indices = sorted_indices[:top_k]
    
    # Convert flattened indices back to (layer, head) coordinates
    top_distracting_heads = []
    for idx in top_distracting_indices:
        layer_idx = idx // num_heads
        head_idx = idx % num_heads
        value = flattened_data[idx]
        top_distracting_heads.append((layer_idx, head_idx, value))
    
    # Function to create a heatmap with consistent styling
    def create_heatmap(ax, data, ylabels, start_layer_idx=0):
        # Create exponential scaling for colors with a larger base for more dramatic differences
        # Apply exponential transformation while preserving signs
        transformed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                transformed_data[i, j] = np.sign(val) * (5 ** abs(val) - 1)

        # Create the heatmap with transformed data for colors but original data for annotations
        h = sns.heatmap(
            transformed_data, 
            ax=ax,
            cmap=cmap,
            vmin=np.min(transformed_data),
            vmax=np.max(transformed_data),
            center=0,
            annot=data,  # Show original values in annotations
            fmt=".3f",
            linewidths=1.5,
            annot_kws={"size": fontsize, "weight": "bold"},
            square=True,
            cbar=False,
            xticklabels=[col.replace('Head_', '') for col in head_columns],
            yticklabels=ylabels
        )
        
        # Add vertical lines to group heads
        for i in range(1, num_heads):
            ax.axvline(i, color='white', linewidth=1.5)
        
        # Add horizontal lines between layers
        for i in range(1, data.shape[0]):
            ax.axhline(i, color='white', linewidth=2.5)
        
        # Invert y-axis to have layer 0 at the top
        ax.invert_yaxis()
        
        # Bigger axis labels
        ax.set_xlabel("Head Index", fontsize=26, fontweight='bold')
        
        # Bigger tick labels
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        return h
    
    # Create the heatmaps for each panel
    heatmaps = []
    for i, (ax, data, labels) in enumerate(zip(axes, panel_data, panel_indices)):
        start_layer_idx = i * layers_per_panel
        h = create_heatmap(ax, data, labels, start_layer_idx)
        heatmaps.append(h)
        
        # Add y-axis label only to the first panel
        if i == 0:
            ax.set_ylabel("Layer Index", fontsize=26, fontweight='bold')
    
    # Add a colorbar to the right side of the figure
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Make room for the colorbar
    cax = plt.axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(heatmaps[0].collections[0], cax=cax)
    
    # Apply inverse transformation to colorbar ticks
    def inverse_transform(x):
        return np.sign(x) * np.log(abs(x) + 1) / np.log(5)
    
    # Get the current ticks and transform them back to original scale
    ticks = cbar.get_ticks()
    new_ticks = [inverse_transform(tick) for tick in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick:.3f}" for tick in new_ticks])
    
    # Define the colorbar label
    colorbar_label = "Distraction Effect: Δ Attention to Instructions"
    
    # Add the label to the colorbar
    cbar.ax.set_ylabel(colorbar_label, fontsize=22, fontweight='bold', labelpad=20)
    cbar.ax.tick_params(labelsize=20)
    
    # Adjust figure layout
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Add a legend for the orange borders
    # legend_text = "Orange borders: Top 20 most distracting heads"
    # plt.figtext(0.5, 0.01, legend_text, ha='center', fontsize=18, fontweight='bold', color='orange')
    
    # Save the figure
    output_path = f"{output_dir}/distraction_effect_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    
    # Create a second visualization focusing on top distracting heads
    # Create a bar chart for top distracting heads
    plt.figure(figsize=(14, 10))
    
    # Prepare data for the bar chart
    head_labels = [f"L{layer}H{head}" for layer, head, _ in top_distracting_heads]
    values = [value for _, _, value in top_distracting_heads]
    
    # Create a gradient color mapping based on values
    colors = []
    for value in values:
        if value < 0:
            # Red with intensity based on how negative
            intensity = min(1.0, abs(value) / abs(vmin))
            colors.append((1.0, 0.5 * (1 - intensity), 0.5 * (1 - intensity)))
        else:
            # Blue with intensity based on how positive
            intensity = min(1.0, value / vmax)
            colors.append((0.5 * (1 - intensity), 0.5 * (1 - intensity), 1.0))
    
    # Create the bar chart with the gradient color
    bars = plt.bar(range(len(head_labels)), values, color=colors, edgecolor='orange', linewidth=2)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        value = values[i]
        plt.text(bar.get_x() + bar.get_width()/2, value - 0.02 if value < 0 else value + 0.02, 
                 f"{value:.3f}", ha='center', va='top' if value < 0 else 'bottom', 
                 fontsize=12, fontweight='bold', color='white' if value < -0.3 else 'black')
    
    # Customize the chart
    plt.xticks(range(len(head_labels)), head_labels, rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Top Distracting Attention Heads", fontsize=24, fontweight='bold')
    plt.ylabel("Distraction Effect", fontsize=18, fontweight='bold')
    plt.xlabel("Layer and Head Index", fontsize=18, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save the figure
    output_path = f"{output_dir}/top_distracting_heads.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Top distracting heads visualization saved to: {output_path}")
    
    return output_path

def create_sample_attention_files(output_dir, num_layers=24, num_heads=12):
    """
    Create sample clean and corrupted attention files for testing.
    
    Args:
        output_dir: Directory to save the sample files
        num_layers: Number of layers to generate
        num_heads: Number of heads per layer
        
    Returns:
        Tuple of (clean_path, corrupted_path)
    """
    clean_path = os.path.join(output_dir, "sample_clean_attention.json")
    corrupted_path = os.path.join(output_dir, "sample_corrupted_attention.json")
    
    # Create random attention patterns
    clean_data = {}
    corrupted_data = {}
    
    # Generate some random but realistic-looking data
    np.random.seed(42)  # For reproducibility
    
    for layer_idx in range(num_layers):
        clean_layer = {}
        corrupted_layer = {}
        layer_key = f"layer_{layer_idx}"
        
        for head_idx in range(num_heads):
            head_key = f"head_{head_idx}"
            
            # For clean, attention to instruction is generally higher
            clean_instr_attn = np.random.uniform(0.3, 0.7)
            
            # For corrupted, attention to instruction is generally lower (distraction)
            # Make certain heads (especially in the middle layers) more susceptible to distraction
            distraction_factor = 0.0
            if layer_idx in [9, 11, 14, 15]:  # Middle layers often show strong distraction
                distraction_factor = np.random.uniform(0.3, 0.9)
            elif layer_idx > 5 and layer_idx < 20:  # Other middle layers show moderate distraction
                distraction_factor = np.random.uniform(0.1, 0.5)
            else:  # Early and late layers often show less distraction
                distraction_factor = np.random.uniform(0.0, 0.2)
                
            # Apply the distraction to reduce the attention to instructions
            corrupted_instr_attn = max(0.0, clean_instr_attn - distraction_factor)
            
            # In a few cases, make the corrupted attention higher than clean (positive effect)
            if np.random.random() < 0.05:  # 5% of heads show increased attention
                corrupted_instr_attn = min(1.0, clean_instr_attn + np.random.uniform(0.05, 0.2))
            
            # Create sample head data
            clean_layer[head_key] = {
                "instruction_attention": float(clean_instr_attn),
                "other_info": "Sample clean attention data"
            }
            
            corrupted_layer[head_key] = {
                "instruction_attention": float(corrupted_instr_attn),
                "other_info": "Sample corrupted attention data"
            }
        
        clean_data[layer_key] = clean_layer
        corrupted_data[layer_key] = corrupted_layer
    
    # Save to files
    with open(clean_path, 'w') as f:
        json.dump(clean_data, f, indent=2)
    
    with open(corrupted_path, 'w') as f:
        json.dump(corrupted_data, f, indent=2)
    
    print(f"Created sample clean attention data: {clean_path}")
    print(f"Created sample corrupted attention data: {corrupted_path}")
    
    return clean_path, corrupted_path

def main():
    """Main function to process data and generate visualizations"""
    parser = argparse.ArgumentParser(description="Analyze and visualize distraction effects in attention mechanisms")
    parser.add_argument("--mode", type=str, choices=["visualize", "analyze", "sample"], default="visualize",
                       help="Mode: 'visualize' existing data, 'analyze' raw attention data, or create 'sample' data")
    parser.add_argument("--csv_path", type=str, default="distraction_analysis/distraction_impacts.csv",
                       help="Path to the distraction effects CSV file (for visualize mode)")
    parser.add_argument("--clean_attn", type=str, default="",
                       help="Path to the clean attention data file (for analyze mode)")
    parser.add_argument("--corrupted_attn", type=str, default="",
                       help="Path to the corrupted attention data file (for analyze mode)")
    parser.add_argument("--num_layers", type=int, default=None,
                       help="Number of layers to analyze")
    parser.add_argument("--num_heads", type=int, default=None,
                       help="Number of heads per layer")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_directory()
    
    if args.mode == "sample":
        # Create sample data
        print("Creating sample attention data...")
        clean_path, corrupted_path = create_sample_attention_files(
            output_dir, 
            num_layers=args.num_layers or 24, 
            num_heads=args.num_heads or 12
        )
        
        # Analyze the sample data
        analyze_distraction_from_raw_data(
            clean_path,
            corrupted_path,
            output_dir,
            args.num_layers,
            args.num_heads
        )
        
    elif args.mode == "analyze":
        # Check if files exist
        if not args.clean_attn or not args.corrupted_attn:
            print("Error: Both clean and corrupted attention file paths must be provided for analyze mode.")
            print("Using --clean_attn and --corrupted_attn arguments.")
            return
            
        if not os.path.exists(args.clean_attn) or not os.path.exists(args.corrupted_attn):
            print(f"Error: One or both attention files not found.")
            return
            
        # Analyze the provided data
        analyze_distraction_from_raw_data(
            args.clean_attn,
            args.corrupted_attn,
            output_dir,
            args.num_layers,
            args.num_heads
        )
        
    else:  # visualize mode
        # Check if CSV file exists
        if not os.path.exists(args.csv_path):
            print(f"Warning: {args.csv_path} not found.")
            user_path = input("Please enter the path to the distraction effects CSV file or press Enter to create sample data: ")
            
            if user_path.strip():
                args.csv_path = user_path
            else:
                print("Creating and using sample data instead...")
                clean_path, corrupted_path = create_sample_attention_files(
                    output_dir,
                    num_layers=args.num_layers or 24,
                    num_heads=args.num_heads or 12
                )
                output_csv = os.path.join(output_dir, "distraction_impacts.csv")
                compute_distraction_effect(
                    clean_path,
                    corrupted_path,
                    output_csv,
                    args.num_layers,
                    args.num_heads
                )
                args.csv_path = output_csv
        
        # Generate the visualization
        visualize_distraction_from_csv(args.csv_path, output_dir)
    
    print("Analysis and visualization completed!")

if __name__ == "__main__":
    main() 