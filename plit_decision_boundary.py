import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from models.classifiers import RandomForestClassifier

def plot_random_forest_decision_boundary(model_path, data_path, output_path=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    
    model = RandomForestClassifier(input_shape=(1,))
    
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from {data_path}")
    
    dataset = torch.load(data_path)
    
    heatmaps = dataset.heatmaps
    labels = dataset.labels
    
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    if not model.is_trained:
        raise ValueError("The model must be trained before plotting the decision boundary")
    
    print("Applying PCA to reduce dimensions for visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(heatmaps.reshape(heatmaps.shape[0], -1))
    
    cmap_light = ListedColormap(['#FFFF99', '#CCFFCC'])
    cmap_bold = ListedColormap(['#CCCC00', '#00CC00'])
    
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    resolution = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    grid_points_original = pca.inverse_transform(grid_points)
    
    Z = model.rf_model.predict_proba(grid_points_original)[:, 1]
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    
    scatter_0 = ax.scatter(X_pca[class_0_mask, 0], X_pca[class_0_mask, 1], 
                          c='#CCCC00', edgecolor='k', s=50, alpha=0.7, marker='o', label='Class 0')
    scatter_1 = ax.scatter(X_pca[class_1_mask, 0], X_pca[class_1_mask, 1], 
                          c='#00CC00', edgecolor='k', s=50, alpha=0.7, marker='s', label='Class 1')
    
    ax.legend(loc="best", title="Classes")
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    
    ax.set_title('Random Forest Decision Boundary (PCA Projection)')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    
    from models.classifiers import RandomForestClassifier
    model = RandomForestClassifier(input_shape=(1,))
    
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from {data_path}")
    
    dataset = torch.load(data_path)
    
    heatmaps = dataset.heatmaps
    labels = dataset.labels
    
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    if not model.is_trained:
        raise ValueError("The model must be trained before plotting the decision boundary")
    
    print("Applying PCA to reduce dimensions to 3D for visualization...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(heatmaps.reshape(heatmaps.shape[0], -1))
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    resolution = 50
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    print("Generating decision surface...")
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    
    z_mean = np.mean(X_pca[:, 2])
    grid_points = np.column_stack([grid_points_2d, np.full(grid_points_2d.shape[0], z_mean)])
    
    grid_points_original = pca.inverse_transform(grid_points)
    
    Z_proba = model.rf_model.predict_proba(grid_points_original)[:, 1]
    Z_proba = Z_proba.reshape(xx.shape)
    
    surf = ax.plot_surface(xx, yy, Z_proba, cmap=plt.cm.coolwarm, alpha=0.6, 
                          linewidth=0, antialiased=True)
    
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    
    ax.scatter(X_pca[class_0_mask, 0], X_pca[class_0_mask, 1], X_pca[class_0_mask, 2], 
              c='blue', edgecolor='k', s=50, alpha=0.7, marker='o', label='Class 0')
    ax.scatter(X_pca[class_1_mask, 0], X_pca[class_1_mask, 1], X_pca[class_1_mask, 2], 
              c='red', edgecolor='k', s=50, alpha=0.7, marker='^', label='Class 1')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Probability of Class 1')
    
    ax.legend(loc="best", title="Classes")
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('Probability / PCA Component 3')
    
    ax.set_title('Random Forest 3D Decision Boundary (PCA Projection)')
    
    if output_path:
        output_path_3d = output_path.replace('.png', '_3d.png')
        plt.savefig(output_path_3d, dpi=300, bbox_inches='tight')
        print(f"3D Plot saved to {output_path_3d}")
    
    plt.show()
    
    return fig

if __name__ == "__main__":
    model_path = "saved_models/attention_classifier.pt"
    
    data_path = "datasets/train.pt"
    
    output_path = "random_forest_decision_boundary.png"
    
    plot_random_forest_decision_boundary(model_path, data_path, output_path)
    
    plot_random_forest_decision_boundary_3d(model_path, data_path, output_path)