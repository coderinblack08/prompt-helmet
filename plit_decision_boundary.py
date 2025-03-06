import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def plot_random_forest_decision_boundary(model_path, data_path, output_path=None):
    """
    Load a RandomForestClassifier model from a specific file path and visualize its decision boundary.
    
    Args:
        model_path: Path to the saved model file (.pt)
        data_path: Path to the data file (.pt) containing heatmaps and labels
        output_path: Path to save the visualization (default: None, will show but not save)
    """
    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    
    # Initialize with dummy input shape, will be overridden by loaded state
    model = RandomForestClassifier(input_shape=(1,))
    
    # Load the state dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # Load the data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from {data_path}")
    
    # Load the dataset
    dataset = torch.load(data_path)
    
    # Extract heatmaps and labels
    heatmaps = dataset.heatmaps
    labels = dataset.labels
    
    # Convert tensors to numpy
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # We need a classifier that has already been fit
    if not model.is_trained:
        raise ValueError("The model must be trained before plotting the decision boundary")
    
    # Use PCA to reduce the heatmap dimensions to 2D for visualization
    print("Applying PCA to reduce dimensions for visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(heatmaps.reshape(heatmaps.shape[0], -1))
    
    # Create color maps
    cmap_light = ListedColormap(['#FFFF99', '#CCFFCC'])
    cmap_bold = ListedColormap(['#CCCC00', '#00CC00'])
    
    # Create a mesh grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    resolution = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Create the feature array for prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Transform grid points back to original space
    grid_points_original = pca.inverse_transform(grid_points)
    
    # Get predictions on the mesh grid
    Z = model.rf_model.predict_proba(grid_points_original)[:, 1]  # Probability of class 1
    Z = Z.reshape(xx.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the decision boundary
    contour = ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    # Plot the training points - create separate scatter plots for each class
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    
    scatter_0 = ax.scatter(X_pca[class_0_mask, 0], X_pca[class_0_mask, 1], 
                          c='#CCCC00', edgecolor='k', s=50, alpha=0.7, marker='o', label='Class 0')
    scatter_1 = ax.scatter(X_pca[class_1_mask, 0], X_pca[class_1_mask, 1], 
                          c='#00CC00', edgecolor='k', s=50, alpha=0.7, marker='s', label='Class 1')
    
    # Add legend
    ax.legend(loc="best", title="Classes")
    
    # Set axis labels
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    
    # Set title
    ax.set_title('Random Forest Decision Boundary (PCA Projection)')
    
    # Set axis limits
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    return fig

def plot_random_forest_decision_boundary_3d(model_path, data_path, output_path=None):
    """
    Load a RandomForestClassifier model from a specific file path and visualize its decision boundary in 3D.
    
    Args:
        model_path: Path to the saved model file (.pt)
        data_path: Path to the data file (.pt) containing heatmaps and labels
        output_path: Path to save the visualization (default: None, will show but not save)
    """
    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    
    # Initialize with dummy input shape, will be overridden by loaded state
    from models.classifiers import RandomForestClassifier
    model = RandomForestClassifier(input_shape=(1,))
    
    # Load the state dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # Load the data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from {data_path}")
    
    # Load the dataset
    dataset = torch.load(data_path)
    
    # Extract heatmaps and labels
    heatmaps = dataset.heatmaps
    labels = dataset.labels
    
    # Convert tensors to numpy
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # We need a classifier that has already been fit
    if not model.is_trained:
        raise ValueError("The model must be trained before plotting the decision boundary")
    
    # Use PCA to reduce the heatmap dimensions to 3D for visualization
    print("Applying PCA to reduce dimensions to 3D for visualization...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(heatmaps.reshape(heatmaps.shape[0], -1))
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a mesh grid for the first two PCA components
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    resolution = 50
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # For each point in the mesh, predict the class
    print("Generating decision surface...")
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    
    # For the third dimension, we'll use the average of the third PCA component
    z_mean = np.mean(X_pca[:, 2])
    grid_points = np.column_stack([grid_points_2d, np.full(grid_points_2d.shape[0], z_mean)])
    
    # Transform grid points back to original space
    grid_points_original = pca.inverse_transform(grid_points)
    
    # Get predictions on the mesh grid
    Z_proba = model.rf_model.predict_proba(grid_points_original)[:, 1]  # Probability of class 1
    Z_proba = Z_proba.reshape(xx.shape)
    
    # Plot the decision surface
    surf = ax.plot_surface(xx, yy, Z_proba, cmap=plt.cm.coolwarm, alpha=0.6, 
                          linewidth=0, antialiased=True)
    
    # Plot the training points
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    
    ax.scatter(X_pca[class_0_mask, 0], X_pca[class_0_mask, 1], X_pca[class_0_mask, 2], 
              c='blue', edgecolor='k', s=50, alpha=0.7, marker='o', label='Class 0')
    ax.scatter(X_pca[class_1_mask, 0], X_pca[class_1_mask, 1], X_pca[class_1_mask, 2], 
              c='red', edgecolor='k', s=50, alpha=0.7, marker='^', label='Class 1')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Probability of Class 1')
    
    # Add legend
    ax.legend(loc="best", title="Classes")
    
    # Set axis labels
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('Probability / PCA Component 3')
    
    # Set title
    ax.set_title('Random Forest 3D Decision Boundary (PCA Projection)')
    
    # Save the plot if output_path is provided
    if output_path:
        output_path_3d = output_path.replace('.png', '_3d.png')
        plt.savefig(output_path_3d, dpi=300, bbox_inches='tight')
        print(f"3D Plot saved to {output_path_3d}")
    
    # Show the plot
    plt.show()
    
    return fig

# Example usage - just modify these paths to point to your model and data
if __name__ == "__main__":
    # Path to your saved RandomForestClassifier model
    model_path = "saved_models/attention_classifier.pt"
    
    # Path to your test data
    data_path = "datasets/train.pt"
    
    # Optional: path to save the visualization
    output_path = "random_forest_decision_boundary.png"
    
    # Generate the 2D visualization
    plot_random_forest_decision_boundary(model_path, data_path, output_path)
    
    # Generate the 3D visualization
    plot_random_forest_decision_boundary_3d(model_path, data_path, output_path)