import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import random
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from models.utils import get_training_and_validation_splits
import umap
from scipy.spatial import ConvexHull
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_model_and_data(model_path, num_samples=500, injection_ratio=0.3):
    """
    Load the SentenceTransformer model and prepare data samples
    with specified ratio of injected examples
    """
    print(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    print("Loading dataset...")
    # Get data using the utility function
    (train_system_prompts, train_user_prompts), _ = get_training_and_validation_splits(total_size=num_samples*2)
    
    # Extract system prompts, user prompts, and labels
    system_prompts = train_system_prompts["system_prompt"].tolist()
    user_prompts = train_user_prompts["user_input"].tolist()
    labels = train_user_prompts["is_injection"].tolist()
    
    # Clean data
    system_prompts = [str(x) if not pd.isna(x) else "" for x in system_prompts]
    user_prompts = [str(x) if not pd.isna(x) else "" for x in user_prompts]
    labels = [int(x) if not pd.isna(x) else 0 for x in labels]
    
    # Create concatenated prompts
    combined_prompts = [
        f"System: {system} User: {user}" 
        for system, user in zip(system_prompts, user_prompts)
    ]
    
    # Separate by class
    benign_indices = [i for i, label in enumerate(labels) if label == 0]
    injection_indices = [i for i, label in enumerate(labels) if label == 1]
    
    # Calculate how many samples to take from each class based on ratio
    num_injection = min(len(injection_indices), int(num_samples * injection_ratio))
    num_benign = min(len(benign_indices), num_samples - num_injection)
    
    print(f"Sampling {num_benign} benign and {num_injection} injection examples (ratio: {injection_ratio:.2f})")
    
    # Sample indices
    sampled_benign = random.sample(benign_indices, num_benign)
    sampled_injection = random.sample(injection_indices, num_injection)
    
    # Combine indices
    sampled_indices = sorted(sampled_benign + sampled_injection)
    
    # Filter data
    combined_prompts = [combined_prompts[i] for i in sampled_indices]
    labels = [labels[i] for i in sampled_indices]
    
    print(f"Using {len(combined_prompts)} samples ({labels.count(0)} benign, {labels.count(1)} injections)")
    
    return model, combined_prompts, labels

def generate_embeddings(model, combined_prompts):
    """
    Generate embeddings for combined prompts
    """
    print("Generating embeddings...")
    embeddings = model.encode(combined_prompts, show_progress_bar=True)
    return embeddings

def custom_metric(x, y, labels_array, weight=5.0):
    """
    Custom distance metric that emphasizes within-class similarity
    """
    # Euclidean distance
    dist = np.sqrt(np.sum((x - y) ** 2))
    
    # If both points are from the same class, reduce the distance
    if labels_array[x[0]] == labels_array[y[0]]:
        dist = dist / weight
    
    return dist

def plot_3d_clusters(embeddings, labels, title, output_path=None, show_plot=False):
    """
    Create a 3D visualization of the embeddings clusters with improved cluster separation
    Stacked vertically with reduced spacing
    """
    # Standardize the embeddings
    print("Standardizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply UMAP with parameters to create tighter clusters
    print("Applying UMAP for 3D visualization (this may take a moment)...")
    # Use more extreme parameters to create tighter clusters
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=5,  # Smaller neighborhood to focus on local structure
        min_dist=0.01,  # Very small min_dist to create tight clusters
        metric='euclidean',
        random_state=42
    )
    embeddings_3d = reducer.fit_transform(embeddings_scaled)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'label': labels
    })
    
    # Create figure with more height for vertical stacking
    fig = plt.figure(figsize=(12, 16))
    
    # Create GridSpec to control subplot spacing
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)  # Reduced hspace to 0.1
    
    # First subplot: 3D scatter with enhanced visuals
    ax1 = fig.add_subplot(gs[0], projection='3d')
    
    # Plot points by class
    class_0_mask = df['label'] == 0
    class_1_mask = df['label'] == 1
    
    # Create separate dataframes for each class
    df_benign = df[class_0_mask]
    df_injection = df[class_1_mask]
    
    # Draw convex hulls around each cluster
    # For benign class (blue)
    if len(df_benign) > 3:
        try:
            # Try to compute a 3D convex hull
            hull = ConvexHull(df_benign[['x', 'y', 'z']].values)
            
            # Get the simplices (triangles) that make up the hull
            for simplex in hull.simplices:
                # Get the vertices of the triangle
                x = df_benign.iloc[simplex, 0].values
                y = df_benign.iloc[simplex, 1].values
                z = df_benign.iloc[simplex, 2].values
                
                # Plot the triangle as a transparent face
                ax1.plot_trisurf(x, y, z, color='blue', alpha=0.1, shade=False)
        except Exception as e:
            print(f"Could not create convex hull for benign class: {e}")
            # Fallback to simple planes
            for z_level in np.linspace(df_benign['z'].min(), df_benign['z'].max(), 5):
                x_grid = np.linspace(df_benign['x'].min() - 0.5, df_benign['x'].max() + 0.5, 20)
                y_grid = np.linspace(df_benign['y'].min() - 0.5, df_benign['y'].max() + 0.5, 20)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = np.full_like(X, z_level)
                ax1.plot_surface(X, Y, Z, color='blue', alpha=0.05, shade=False)
    
    # For injection class (red)
    if len(df_injection) > 3:
        try:
            # Try to compute a 3D convex hull
            hull = ConvexHull(df_injection[['x', 'y', 'z']].values)
            
            # Get the simplices (triangles) that make up the hull
            for simplex in hull.simplices:
                # Get the vertices of the triangle
                x = df_injection.iloc[simplex, 0].values
                y = df_injection.iloc[simplex, 1].values
                z = df_injection.iloc[simplex, 2].values
                
                # Plot the triangle as a transparent face
                ax1.plot_trisurf(x, y, z, color='red', alpha=0.1, shade=False)
        except Exception as e:
            print(f"Could not create convex hull for injection class: {e}")
            # Fallback to simple planes
            for z_level in np.linspace(df_injection['z'].min(), df_injection['z'].max(), 5):
                x_grid = np.linspace(df_injection['x'].min() - 0.5, df_injection['x'].max() + 0.5, 20)
                y_grid = np.linspace(df_injection['y'].min() - 0.5, df_injection['y'].max() + 0.5, 20)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = np.full_like(X, z_level)
                ax1.plot_surface(X, Y, Z, color='red', alpha=0.05, shade=False)
    
    # Plot injections first (red) with larger markers
    ax1.scatter(
        df.loc[class_1_mask, 'x'], 
        df.loc[class_1_mask, 'y'], 
        df.loc[class_1_mask, 'z'],
        c='red', edgecolor='black', s=150, alpha=1.0, marker='^', label='Injection',
        zorder=10  # Higher zorder to ensure they're drawn on top
    )
    
    # Then plot benign examples (blue) with smaller, more transparent markers
    ax1.scatter(
        df.loc[class_0_mask, 'x'], 
        df.loc[class_0_mask, 'y'], 
        df.loc[class_0_mask, 'z'],
        c='blue', edgecolor='k', s=50, alpha=0.4, marker='o', label='Benign',
        zorder=5
    )
    
    # Add legend
    ax1.legend(loc="upper right", title="Classes", fontsize=12)
    
    # Set axis labels
    ax1.set_xlabel('UMAP Component 1', fontsize=14)
    ax1.set_ylabel('UMAP Component 2', fontsize=14)
    ax1.set_zlabel('UMAP Component 3', fontsize=14)
    
    # Set title with less padding
    ax1.set_title(f'3D Cluster Visualization: {title}', fontsize=16, pad=10)
    
    # Adjust view angle to better show separation
    ax1.view_init(elev=25, azim=120)
    
    # Second subplot: 2D projection with contours
    ax2 = fig.add_subplot(gs[1])
    
    # Create a 2D UMAP projection with even more extreme parameters
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.01,
        metric='euclidean',
        random_state=42
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings_scaled)
    
    df_2d = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    # Create a mesh grid for the 2D plot
    x_min, x_max = df_2d['x'].min() - 1, df_2d['x'].max() + 1
    y_min, y_max = df_2d['y'].min() - 1, df_2d['y'].max() + 1
    
    # Create a 2D kernel density estimate for each class
    for label, color, marker, name in [(0, 'blue', 'o', 'Benign'), (1, 'red', '^', 'Injection')]:
        mask = df_2d['label'] == label
        if sum(mask) > 2:  # Need at least 3 points for KDE
            sns.kdeplot(
                x=df_2d.loc[mask, 'x'], 
                y=df_2d.loc[mask, 'y'],
                ax=ax2,
                fill=True,
                alpha=0.3,
                levels=5,
                color=color,
                label=name
            )
    
    # Plot the points - injections first with larger markers
    ax2.scatter(
        df_2d.loc[class_1_mask, 'x'], 
        df_2d.loc[class_1_mask, 'y'],
        c='red', edgecolor='black', s=150, alpha=1.0, marker='^',
        zorder=10
    )
    ax2.scatter(
        df_2d.loc[class_0_mask, 'x'], 
        df_2d.loc[class_0_mask, 'y'],
        c='blue', edgecolor='k', s=50, alpha=0.4, marker='o',
        zorder=5
    )
    
    # Add legend
    ax2.legend(loc="best", title="Classes")
    
    # Set axis labels
    ax2.set_xlabel('UMAP Component 1', fontsize=14)
    ax2.set_ylabel('UMAP Component 2', fontsize=14)
    
    # Set title with less padding
    ax2.set_title(f'2D Projection with Density Contours: {title}', fontsize=16, pad=10)
    
    # Set axis limits
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout - don't use tight_layout since we're using GridSpec
    plt.subplots_adjust(top=0.95, bottom=0.05)  # Adjust top and bottom margins
    
    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
def main():
    # Path to the model
    model_path = "./saved_models/embeddings_model_all-MiniLM-L6-v2.bin"
    
    # Number of samples to use
    num_samples = 500
    
    # Injection ratio (30% injected, 70% benign)
    injection_ratio = 0.3
    
    # Load model and data
    model, combined_prompts, labels = load_model_and_data(
        model_path, 
        num_samples=num_samples,
        injection_ratio=injection_ratio
    )
    
    # Generate embeddings
    embeddings = generate_embeddings(model, combined_prompts)
    
    # Create output directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Visualize embeddings
    plot_3d_clusters(
        embeddings, 
        labels, 
        "Prompt Embeddings",
        "visualizations/combined_embeddings_3d_enhanced.png",
        show_plot=True  # Set to False if running headless
    )
    
    print("Visualization completed!")

if __name__ == "__main__":
    main()