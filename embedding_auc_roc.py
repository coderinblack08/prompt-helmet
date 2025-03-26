import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
from tqdm import tqdm
from models.utils import get_training_and_validation_splits

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_model_and_data(model_path, num_samples=None):
    """
    Load the SentenceTransformer model and prepare data
    """
    print(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    print("Loading dataset...")
    # Get data using the utility function
    (_, _), (val_system_prompts, val_user_prompts) = get_training_and_validation_splits(total_size=num_samples)
    
    if num_samples is not None:
        # Limit the number of samples if specified
        val_system_prompts = val_system_prompts.head(num_samples)
        val_user_prompts = val_user_prompts.head(num_samples)
    
    # Clean data
    system_prompts = [str(x) if not pd.isna(x) else "" for x in val_system_prompts["system_prompt"].tolist()]
    user_prompts = [str(x) if not pd.isna(x) else "" for x in val_user_prompts["user_input"].tolist()]
    labels = [int(x) if not pd.isna(x) else 0 for x in val_user_prompts["is_injection"].tolist()]
    
    print(f"Using {len(system_prompts)} samples ({labels.count(0)} benign, {labels.count(1)} injections)")
    
    return model, system_prompts, user_prompts, labels

def plot_roc_curve(similarities, true_labels, model_name, output_path=None, show_plot=True):
    """
    Plot ROC curve for the embedding model using cosine similarity scores
    """
    print("Generating ROC curve...")
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(true_labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(
        fpr, tpr,
        label=f'Cosine Similarity (AUC = {roc_auc:.3f})',
        color='blue', lw=2
    )
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8, label='Random Chance')
    
    # Set labels and title
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Embedding Model: {model_name}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    
    # Set axes limits
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    
    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return roc_auc

def plot_precision_recall_curve(similarities, true_labels, model_name, output_path=None, show_plot=True):
    """
    Plot Precision-Recall curve for the embedding model using cosine similarity scores
    """
    print("Generating Precision-Recall curve...")
    plt.figure(figsize=(10, 8))
    
    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels, similarities)
    ap = average_precision_score(true_labels, similarities)
    
    # Plot Precision-Recall curve
    plt.plot(
        recall, precision,
        label=f'Cosine Similarity (AP = {ap:.3f})',
        color='red', lw=2
    )
    
    # Plot random chance line (depends on class balance)
    no_skill = sum(true_labels) / len(true_labels)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', alpha=0.8, label='Random Chance')
    
    # Set labels and title
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - Embedding Model: {model_name}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    
    # Set axes limits
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    
    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"PR plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return ap

def plot_similarity_threshold_metrics(similarities, true_labels, model_name, output_path=None, show_plot=True):
    """
    Plot metrics at different similarity thresholds
    """
    print("Calculating metrics at different thresholds...")
    thresholds = np.arange(-0.2, 1.0, 0.02)
    
    # Calculate metrics for each threshold
    metrics = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for threshold in thresholds:
        # Invert predictions since higher similarity should indicate benign examples (opposite of what we want)
        predictions = (similarities < threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((predictions == 1) & (true_labels == 1))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['threshold'].append(threshold)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
    
    # Find best F1 threshold
    best_f1_idx = np.argmax(metrics['f1'])
    best_threshold = metrics['threshold'][best_f1_idx]
    best_f1 = metrics['f1'][best_f1_idx]
    
    print(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    
    plt.plot(metrics['threshold'], metrics['accuracy'], label='Accuracy', color='blue')
    plt.plot(metrics['threshold'], metrics['precision'], label='Precision', color='green')
    plt.plot(metrics['threshold'], metrics['recall'], label='Recall', color='red')
    plt.plot(metrics['threshold'], metrics['f1'], label='F1 Score', color='purple', linewidth=2)
    
    # Mark the best threshold
    plt.axvline(x=best_threshold, color='black', linestyle='--', alpha=0.7)
    plt.text(best_threshold + 0.02, 0.5, f'Best Threshold: {best_threshold:.2f}', bbox=dict(facecolor='white', alpha=0.8))
    
    # Set labels and title
    plt.xlabel('Similarity Threshold', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.title(f'Metrics vs. Similarity Threshold - {model_name}', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="best", fontsize=10)
    
    # Set axes limits
    plt.xlim([min(thresholds) - 0.05, max(thresholds) + 0.05])
    plt.ylim([-0.05, 1.05])
    
    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Threshold metrics plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return best_threshold, best_f1

def main():
    # Path to the model
    model_path = "./saved_models/embeddings_model_all-MiniLM-L6-v2.bin"
    model_name = "all-MiniLM-L6-v2"
    
    # Number of samples to use (None for all available samples)
    num_samples = None
    
    # Load model and data
    model, system_prompts, user_prompts, labels = load_model_and_data(
        model_path,
        num_samples=num_samples
    )
    
    print("Generating embeddings...")
    # Generate embeddings for system and user prompts
    system_embeddings = model.encode(system_prompts, show_progress_bar=True)
    user_embeddings = model.encode(user_prompts, show_progress_bar=True)
    
    # Compute cosine similarities
    similarities = cosine_similarity(system_embeddings, user_embeddings)
    # We take the diagonal which represents similarity between corresponding system-user prompt pairs
    similarity_scores = similarities.diagonal()
    
    # Create output directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Plot ROC curve
    plot_roc_curve(
        similarity_scores,
        labels,
        model_name,
        "visualizations/embedding_roc_curve.png",
        show_plot=True
    )
    
    # Plot Precision-Recall curve
    plot_precision_recall_curve(
        similarity_scores,
        labels,
        model_name,
        "visualizations/embedding_precision_recall_curve.png",
        show_plot=True
    )
    
    # Plot metrics at different thresholds
    plot_similarity_threshold_metrics(
        similarity_scores,
        labels,
        model_name,
        "visualizations/embedding_threshold_metrics.png",
        show_plot=True
    )
    
    print("AUC-ROC analysis completed!")

if __name__ == "__main__":
    main() 