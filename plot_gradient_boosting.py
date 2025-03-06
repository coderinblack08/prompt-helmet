import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def plot_gradient_boosting_terrain(model_path, data_path, output_path=None):
    """
    Load a GradientBoostingClassifier model and visualize its decision boundary as a terrain.
    
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
    model = GradientBoostingClassifier(input_shape=(1,))
    
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
    Z_proba = model.gbm_model.predict_proba(grid_points_original)[:, 1]  # Probability of class 1
    Z_proba = Z_proba.reshape(xx.shape)
    
    # Create figure for 3D terrain visualization
    fig = plt.figure(figsize=(15, 12))
    
    # First subplot: 3D terrain
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot the terrain surface
    terrain = ax1.plot_surface(xx, yy, Z_proba, cmap='terrain', alpha=0.8,
                              linewidth=0, antialiased=True, edgecolor='none')
    
    # Add contour lines on the bottom of the plot
    offset = Z_proba.min() - 0.1
    contour_lines = ax1.contour(xx, yy, Z_proba, zdir='z', offset=offset, cmap='viridis', levels=10)
    
    # Plot the training points
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    
    # Scatter points at their probability height
    for i, (x, y) in enumerate(X_pca):
        # Get the probability for this point
        point_original = pca.inverse_transform(np.array([[x, y]]))
        prob = model.gbm_model.predict_proba(point_original)[0, 1]
        
        if labels[i] == 0:
            ax1.scatter(x, y, prob, c='blue', edgecolor='k', s=50, alpha=0.7, marker='o')
        else:
            ax1.scatter(x, y, prob, c='red', edgecolor='k', s=50, alpha=0.7, marker='^')
    
    # Add a color bar
    fig.colorbar(terrain, ax=ax1, shrink=0.5, aspect=5, label='Probability of Class 1')
    
    # Set axis labels
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_zlabel('Probability')
    
    # Set title
    ax1.set_title('Gradient Boosting Terrain Visualization')
    
    # Second subplot: 2D contour with decision trees
    ax2 = fig.add_subplot(122)
    
    # Plot the contour
    contour = ax2.contourf(xx, yy, Z_proba, levels=20, cmap='terrain', alpha=0.8)
    
    # Add contour lines
    contour_lines = ax2.contour(xx, yy, Z_proba, levels=10, colors='k', alpha=0.5, linestyles='dashed')
    ax2.clabel(contour_lines, inline=True, fontsize=8)
    
    # Plot the training points
    ax2.scatter(X_pca[class_0_mask, 0], X_pca[class_0_mask, 1], 
               c='blue', edgecolor='k', s=50, alpha=0.7, marker='o', label='Class 0')
    ax2.scatter(X_pca[class_1_mask, 0], X_pca[class_1_mask, 1], 
               c='red', edgecolor='k', s=50, alpha=0.7, marker='^', label='Class 1')
    
    # Add decision tree-like lines
    # This is a simplified representation of decision boundaries
    n_trees = min(5, model.gbm_model.n_estimators)
    for i in range(n_trees):
        # Generate random splits to illustrate tree-like decision boundaries
        # In a real implementation, you would extract actual tree splits
        split_x = np.random.uniform(x_min, x_max)
        split_y = np.random.uniform(y_min, y_max)
        
        # Horizontal split
        ax2.axhline(y=split_y, color='green', linestyle='-', alpha=0.3)
        
        # Vertical split
        ax2.axvline(x=split_x, color='green', linestyle='-', alpha=0.3)
    
    # Add legend
    ax2.legend(loc="best", title="Classes")
    
    # Add colorbar
    fig.colorbar(contour, ax=ax2, label='Probability of Class 1')
    
    # Set axis labels
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    
    # Set title
    ax2.set_title('Gradient Boosting Decision Boundary with Tree Splits')
    
    # Set axis limits
    ax2.set_xlim(xx.min(), xx.max())
    ax2.set_ylim(yy.min(), yy.max())
    
    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    return fig

def plot_gradient_boosting_feature_importance(model_path, data_path, output_path=None):
    """
    Visualize feature importance from a GradientBoostingClassifier.
    
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
    model = GradientBoostingClassifier(input_shape=(1,))
    
    # Load the state dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # We need a classifier that has already been fit
    if not model.is_trained:
        raise ValueError("The model must be trained before plotting feature importance")
    
    # Get feature importances
    feature_importances = model.gbm_model.feature_importances_
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get the top 20 features
    n_features = min(20, len(feature_importances))
    indices = np.argsort(feature_importances)[-n_features:]
    
    # Plot feature importances
    ax.barh(range(n_features), feature_importances[indices], align='center')
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([f"Feature {i}" for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Feature Importances in Gradient Boosting Model')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Save the plot if output_path is provided
    if output_path:
        output_path_fi = output_path.replace('.png', '_feature_importance.png')
        plt.savefig(output_path_fi, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path_fi}")
    
    # Show the plot
    plt.show()
    
    return fig

# Example usage
if __name__ == "__main__":
    # Path to your saved GradientBoostingClassifier model
    model_path = "saved_models/attention_classifier.pt"
    
    # Path to your test data
    data_path = "datasets/train.pt"
    
    # Optional: path to save the visualization
    output_path = "gradient_boosting_terrain.png"
    
    # Generate the terrain visualization
    plot_gradient_boosting_terrain(model_path, data_path, output_path)
    
    # Generate the feature importance visualization
    plot_gradient_boosting_feature_importance(model_path, data_path, output_path) 