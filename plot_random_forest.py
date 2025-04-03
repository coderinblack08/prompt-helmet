import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from models.classifiers import RandomForestClassifier
import torch
import os

def plot_random_forest_decision_boundary(model_path, data_path, output_path=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    
    model = RandomForestClassifier(input_shape=(2,))
    
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    print(f"Loading data from {data_path}")
    try:
        if data_path.endswith('.pt'):
            data = torch.load(data_path)
            
            if isinstance(data, dict) and 'X' in data and 'y' in data:
                X = data['X']
                y = data['y']
            elif isinstance(data, tuple) and len(data) == 2:
                X, y = data
            elif hasattr(data, 'heatmaps') and hasattr(data, 'labels'):
                X = data.heatmaps
                y = data.labels
            else:
                print(f"Unexpected data format. Data type: {type(data)}")
                print(f"Data content: {data}")
                raise ValueError("Unsupported data format")
                
        elif data_path.endswith('.npz'):
            data = np.load(data_path)
            X = torch.tensor(data['X'], dtype=torch.float32)
            y = torch.tensor(data['y'], dtype=torch.long)
        else:
            raise ValueError(f"Unsupported data file format: {data_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Attempting to load data as a dataset object...")
        
        try:
            dataset = torch.load(data_path)
            X = []
            y = []
            for i in range(len(dataset)):
                features, label = dataset[i]
                X.append(features)
                y.append(label)
            X = torch.stack(X)
            y = torch.tensor(y)
            print(f"Successfully loaded dataset with {len(X)} samples")
        except Exception as e2:
            print(f"Failed to load as dataset: {e2}")
            raise ValueError(f"Could not load data: {e}, {e2}")
    
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    
    if not model.is_trained:
        raise ValueError("The model must be trained before plotting the decision boundary")
    
    if X.shape[1] > 2:
        print(f"Using only the first 2 features out of {X.shape[1]} for visualization")
        X = X[:, :2]
    
    cmap_light = ListedColormap(['#FFFF99', '#CCFFCC'])
    cmap_bold = ListedColormap(['#CCCC00', '#00CC00'])
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    resolution = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    if model.rf_model.n_features_in_ > 2:
        print(f"Model expects {model.rf_model.n_features_in_} features, padding with zeros")
        grid_points = np.hstack([grid_points, np.zeros((grid_points.shape[0], model.rf_model.n_features_in_ - 2))])
    
    print("Generating decision boundary...")
    Z = model.rf_model.predict_proba(grid_points)[:, 1]
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, 
                         edgecolor='k', s=50, alpha=0.7,
                         marker=np.where(y == 0, 'o', 's'))
    
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="best", title="Classes")
    ax.add_artist(legend1)
    
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    
    ax.set_title('Random Forest Decision Boundary')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()
    
    return fig

if __name__ == "__main__":
    model_path = "path/to/your/random_forest_model.pt"
    
    data_path = "path/to/your/test_data.pt"
    
    output_path = "random_forest_decision_boundary.png"
    
    plot_random_forest_decision_boundary(model_path, data_path, output_path) 