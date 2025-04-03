import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from models.classifiers import GradientBoostingClassifier

def plot_gradient_boosting_terrain(model_path, data_path, output_path=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    
    model = GradientBoostingClassifier(input_shape=(1,))
    
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
    
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    resolution = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    grid_points_original = pca.inverse_transform(grid_points)
    
    Z_proba = model.gbm_model.predict_proba(grid_points_original)[:, 1]
    Z_proba = Z_proba.reshape(xx.shape)
    
    fig = plt.figure(figsize=(15, 12))
    
    ax1 = fig.add_subplot(121, projection='3d')
    
    terrain = ax1.plot_surface(xx, yy, Z_proba, cmap='terrain', alpha=0.8,
                              linewidth=0, antialiased=True, edgecolor='none')
    
    offset = Z_proba.min() - 0.1
    contour_lines = ax1.contour(xx, yy, Z_proba, zdir='z', offset=offset, cmap='viridis', levels=10)
    
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    
    for i, (x, y) in enumerate(X_pca):
        point_original = pca.inverse_transform(np.array([[x, y]]))
        prob = model.gbm_model.predict_proba(point_original)[0, 1]
        
        if labels[i] == 0:
            ax1.scatter(x, y, prob, c='blue', edgecolor='k', s=50, alpha=0.7, marker='o')
        else:
            ax1.scatter(x, y, prob, c='red', edgecolor='k', s=50, alpha=0.7, marker='^')
    
    fig.colorbar(terrain, ax=ax1, shrink=0.5, aspect=5, label='Probability of Class 1')
    
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_zlabel('Probability')
    
    ax1.set_title('Gradient Boosting Terrain Visualization')
    
    ax2 = fig.add_subplot(122)
    
    contour = ax2.contourf(xx, yy, Z_proba, levels=20, cmap='terrain', alpha=0.8)
    
    contour_lines = ax2.contour(xx, yy, Z_proba, levels=10, colors='k', alpha=0.5, linestyles='dashed')
    ax2.clabel(contour_lines, inline=True, fontsize=8)
    
    ax2.scatter(X_pca[class_0_mask, 0], X_pca[class_0_mask, 1], 
               c='blue', edgecolor='k', s=50, alpha=0.7, marker='o', label='Class 0')
    ax2.scatter(X_pca[class_1_mask, 0], X_pca[class_1_mask, 1], 
               c='red', edgecolor='k', s=50, alpha=0.7, marker='^', label='Class 1')
    
    n_trees = min(5, model.gbm_model.n_estimators)
    for i in range(n_trees):
        split_x = np.random.uniform(x_min, x_max)
        split_y = np.random.uniform(y_min, y_max)
        
        ax2.axhline(y=split_y, color='green', linestyle='-', alpha=0.3)
        
        ax2.axvline(x=split_x, color='green', linestyle='-', alpha=0.3)
    
    ax2.legend(loc="best", title="Classes")
    
    fig.colorbar(contour, ax=ax2, label='Probability of Class 1')
    
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    
    ax2.set_title('Gradient Boosting Decision Boundary with Tree Splits')
    
    ax2.set_xlim(xx.min(), xx.max())
    ax2.set_ylim(yy.min(), yy.max())
    
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    
    model = GradientBoostingClassifier(input_shape=(1,))
    
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    if not model.is_trained:
        raise ValueError("The model must be trained before plotting feature importance")
    
    feature_importances = model.gbm_model.feature_importances_
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_features = min(20, len(feature_importances))
    indices = np.argsort(feature_importances)[-n_features:]
    
    ax.barh(range(n_features), feature_importances[indices], align='center')
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([f"Feature {i}" for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Feature Importances in Gradient Boosting Model')
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if output_path:
        output_path_fi = output_path.replace('.png', '_feature_importance.png')
        plt.savefig(output_path_fi, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path_fi}")
    
    plt.show()
    
    return fig

if __name__ == "__main__":
    model_path = "saved_models/attention_classifier.pt"
    
    data_path = "datasets/train.pt"
    
    output_path = "gradient_boosting_terrain.png"
    
    plot_gradient_boosting_terrain(model_path, data_path, output_path)
    
    plot_gradient_boosting_feature_importance(model_path, data_path, output_path) 