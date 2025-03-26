import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from models.utils import get_training_and_validation_splits

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def find_optimal_threshold(model, val_system_prompts, val_user_prompts, threshold_range=None):
    """
    Find the optimal threshold for classification based on F1 score
    """
    if threshold_range is None:
        threshold_range = np.arange(-0.2, 1.0, 0.1)
        
    system_prompt_dict = {}
    system_embeddings = []
    for prompt in val_system_prompts["system_prompt"].tolist():
        if prompt not in system_prompt_dict:
            system_prompt_dict[prompt] = torch.from_numpy(model.encode(prompt))
        system_embeddings.append(system_prompt_dict[prompt])

    system_embeddings = torch.stack(system_embeddings)
    user_embeddings = torch.from_numpy(model.encode(val_user_prompts["user_input"].tolist()))
    
    system_embeddings = system_embeddings.cpu().numpy()
    user_embeddings = user_embeddings.cpu().numpy()
    
    similarities = cosine_similarity(system_embeddings, user_embeddings)
    true_labels = val_user_prompts["is_injection"].tolist()
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in threshold_range:
        predictions = (similarities.diagonal() > threshold).astype(int)
        f1 = f1_score(true_labels, predictions)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1, similarities, true_labels

def load_model_and_data(model_path, num_samples=100):
    """
    Load the SentenceTransformer model and prepare data samples
    """
    print(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    print("Loading dataset...")
    # Get data using the utility function
    (train_system_prompts, train_user_prompts), (val_system_prompts, val_user_prompts) = get_training_and_validation_splits(total_size=num_samples*2)
    
    return model, val_system_prompts, val_user_prompts

def generate_2d_embeddings(model, system_prompts, user_prompts):
    """
    Generate 2D embeddings for visualization
    """
    # For simplicity, we'll use PCA to reduce to 2D directly
    # In a real scenario, you might want to use t-SNE or UMAP
    from sklearn.decomposition import PCA
    
    # Get the original embeddings
    system_prompts_list = system_prompts["system_prompt"].tolist()
    user_prompts_list = user_prompts["user_input"].tolist()
    
    # Clean data
    system_prompts_list = [str(x) if not pd.isna(x) else "" for x in system_prompts_list]
    user_prompts_list = [str(x) if not pd.isna(x) else "" for x in user_prompts_list]
    
    # Create concatenated prompts
    combined_prompts = [
        f"System: {system} User: {user}" 
        for system, user in zip(system_prompts_list, user_prompts_list)
    ]
    
    # Generate embeddings for combined prompts
    embeddings = model.encode(combined_prompts)
    
    # Fit PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    return embeddings_2d, pca

def plot_2d_inference(embeddings_2d, labels, similarities, threshold, test_idx=None):
    """
    Create a 2D plot showing the embeddings and cosine similarity
    """
    plt.figure(figsize=(14, 12))
    
    # If no test index is provided, randomly select one
    if test_idx is None:
        test_idx = random.randint(0, len(labels) - 1)
    
    # Extract the test point
    test_point = embeddings_2d[test_idx]
    test_label = labels[test_idx]
    test_sim = similarities[test_idx, test_idx]
    
    # Separate points by class
    benign_indices = [i for i, label in enumerate(labels) if label == 0]
    injection_indices = [i for i, label in enumerate(labels) if label == 1]
    
    # Plot injections first (red) with larger markers
    plt.scatter(embeddings_2d[injection_indices, 0], embeddings_2d[injection_indices, 1], 
                c='red', marker='^', s=150, alpha=0.8, edgecolor='black', label='Injection Prompts')
    
    # Then plot benign examples (blue)
    plt.scatter(embeddings_2d[benign_indices, 0], embeddings_2d[benign_indices, 1], 
                c='blue', marker='^', s=100, alpha=0.7, edgecolor='black', label='Benign Prompts')
    
    # Plot the test point with a star
    plt.scatter(test_point[0], test_point[1], c='green', marker='*', s=400, 
                edgecolor='black', linewidth=2, label='Test Prompt')
    
    # Add a text annotation with the cosine similarity
    plt.annotate(f"Sim: {test_sim:.3f}\nThreshold: {threshold:.3f}\nPrediction: {'Injection' if test_sim > threshold else 'Benign'}\nTrue: {'Injection' if test_label == 1 else 'Benign'}", 
                 (test_point[0] + 0.1, test_point[1] + 0.1), fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8))
    
    # Find and draw connections to N nearest neighbors
    n_neighbors = 3
    
    # Calculate distances from test point to all other points
    distances = np.sqrt(np.sum((embeddings_2d - test_point)**2, axis=1))
    # Exclude the test point itself
    distances[test_idx] = np.inf
    
    # Get indices of N nearest neighbors
    nearest_indices = np.argsort(distances)[:n_neighbors]
    
    # Draw connections and add similarity scores
    for idx in nearest_indices:
        # Calculate cosine similarity
        sim = similarities[test_idx, idx]
        
        # Draw line to neighbor
        plt.plot([test_point[0], embeddings_2d[idx, 0]], 
                 [test_point[1], embeddings_2d[idx, 1]], 
                 'k--', alpha=0.6, linewidth=1)
        
        # Add similarity score
        mid_x = (test_point[0] + embeddings_2d[idx, 0]) / 2
        mid_y = (test_point[1] + embeddings_2d[idx, 1]) / 2
        plt.annotate(f"{sim:.3f}", (mid_x, mid_y), fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    
    # Add title and legend
    plt.title("2D Visualization of Embedding Inference with Cosine Similarity", fontsize=16)
    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanation text
    explanation = (
        "Explanation:\n"
        "1. System and user prompts are concatenated and embedded in 2D space\n"
        "2. Cosine similarity is calculated between paired prompts\n"
        "3. If similarity > threshold, the prompt is classified as an injection\n"
        "4. The optimal threshold is found by maximizing F1 score on validation data\n"
        "5. Lines to nearest neighbors show similarity scores between embeddings"
    )
    plt.figtext(0.02, 0.02, explanation, fontsize=20, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
    
    # Extreme zoom on the test point (10x closer)
    margin = 0.004  # Ultra-tight zoom (10x closer than before)
    plt.xlim(test_point[0] - margin, test_point[0] + margin)
    plt.ylim(test_point[1] - margin, test_point[1] + margin)
    
    plt.tight_layout()
    plt.savefig("embedding_inference_2d_extreme_zoom.png", dpi=300)
    plt.show()

def main():
    # Path to the model
    model_path = "./saved_models/embeddings_model_all-MiniLM-L6-v2.bin"
    
    # Load model and data
    model, val_system_prompts, val_user_prompts = load_model_and_data(model_path)
    
    # Find optimal threshold
    threshold, f1, similarities, true_labels = find_optimal_threshold(
        model, val_system_prompts, val_user_prompts
    )
    
    print(f"Optimal threshold: {threshold:.3f} (F1: {f1:.3f})")
    
    # Generate 2D embeddings
    embeddings_2d, _ = generate_2d_embeddings(
        model, val_system_prompts, val_user_prompts
    )
    
    # Find a random injection example for visualization
    injection_indices = [i for i, label in enumerate(true_labels) if label == 1]
    if injection_indices:
        test_idx = random.choice(injection_indices)
    else:
        test_idx = random.randint(0, len(true_labels) - 1)
    
    # Plot the 2D visualization
    plot_2d_inference(
        embeddings_2d, 
        true_labels, 
        similarities, 
        threshold,
        test_idx=test_idx
    )

if __name__ == "__main__":
    main() 