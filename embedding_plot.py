import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import plotly.express as px
import pandas as pd
from tqdm import tqdm
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load the fine-tuned model
model_path = "embeddings_model_all_minilm"  # Update this path if needed
model = SentenceTransformer(model_path)
print(f"Loaded model: {model_path}")

# Load dataset - adjust the path/name as needed
print("Loading dataset...")
dataset = load_dataset("your_dataset_name_or_path")  # Replace with your actual dataset

# Assuming your dataset has a 'train' split and contains 'text' and 'label' fields
# where 'label' is 0 for benign and 1 for non-benign (adjust as needed)
train_data = dataset["train"]

# Extract benign and non-benign examples
benign_examples = [item["text"] for item in train_data if item["label"] == 0]
non_benign_examples = [item["text"] for item in train_data if item["label"] == 1]

print(f"Found {len(benign_examples)} benign examples and {len(non_benign_examples)} non-benign examples")

# Sample a balanced subset if there are too many examples
max_examples_per_class = 500  # Adjust based on your computational resources
if len(benign_examples) > max_examples_per_class:
    benign_examples = random.sample(benign_examples, max_examples_per_class)
if len(non_benign_examples) > max_examples_per_class:
    non_benign_examples = random.sample(non_benign_examples, max_examples_per_class)

# Combine examples and create labels
texts = benign_examples + non_benign_examples
labels = [0] * len(benign_examples) + [1] * len(non_benign_examples)
label_names = ["Benign", "Non-Benign"]

print(f"Using {len(texts)} examples for visualization ({len(benign_examples)} benign, {len(non_benign_examples)} non-benign)")

# Generate embeddings
print("Generating embeddings...")
embeddings = []
batch_size = 32

# Process in batches for efficiency
for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    batch_embeddings = model.encode(batch, convert_to_tensor=True)
    embeddings.append(batch_embeddings)

# Concatenate all embeddings
if len(embeddings) > 1:
    embeddings = torch.cat(embeddings, dim=0)
else:
    embeddings = embeddings[0]

# Convert to numpy for sklearn
embeddings_np = embeddings.cpu().numpy()
print(f"Generated {len(embeddings_np)} embeddings of dimension {embeddings_np.shape[1]}")

# Reduce dimensionality to 3D for visualization
print("Reducing dimensionality with t-SNE to 3D...")
tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(texts)-1))
embeddings_3d = tsne.fit_transform(embeddings_np)

# Create short labels for the plot
short_texts = [txt[:30] + "..." if len(txt) > 30 else txt for txt in texts]

# Create a DataFrame for Plotly
df = pd.DataFrame({
    'x': embeddings_3d[:, 0],
    'y': embeddings_3d[:, 1],
    'z': embeddings_3d[:, 2],
    'label': [label_names[l] for l in labels],
    'text': texts,
    'short_text': short_texts
})

# Create interactive 3D plot with Plotly
fig = px.scatter_3d(
    df, x='x', y='y', z='z',
    color='label', 
    color_discrete_map={"Benign": "blue", "Non-Benign": "red"},
    hover_data=['text'],
    text='short_text',
    title='3D Visualization of Text Embeddings by Class',
    labels={'label': 'Class'}
)

# Update marker size and opacity
fig.update_traces(marker=dict(size=5, opacity=0.7))

# Add text labels
fig.update_traces(
    hovertemplate='<b>%{text}</b><br>Class: %{marker.color}<extra></extra>'
)

# Improve layout
fig.update_layout(
    scene=dict(
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        zaxis_title='t-SNE Dimension 3'
    ),
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=30)
)

# Save as HTML for interactive viewing
fig.write_html('embedding_visualization_3d.html')
print("Saved interactive 3D visualization to 'embedding_visualization_3d.html'")

# Also create a static 3D plot with matplotlib
fig_static = plt.figure(figsize=(12, 10))
ax = fig_static.add_subplot(111, projection='3d')

# Plot benign examples in blue
benign_points = embeddings_3d[:len(benign_examples)]
ax.scatter(
    benign_points[:, 0], 
    benign_points[:, 1], 
    benign_points[:, 2],
    color='blue',
    label='Benign',
    alpha=0.7,
    s=30
)

# Plot non-benign examples in red
non_benign_points = embeddings_3d[len(benign_examples):]
ax.scatter(
    non_benign_points[:, 0], 
    non_benign_points[:, 1], 
    non_benign_points[:, 2],
    color='red',
    label='Non-Benign',
    alpha=0.7,
    s=30
)

# Add labels and title
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')
ax.set_title('3D Visualization of Text Embeddings by Class')
ax.legend()

# Save the static plot
plt.tight_layout()
plt.savefig('embedding_visualization_3d_static.png', dpi=300)
print("Saved static 3D visualization to 'embedding_visualization_3d_static.png'")

# Calculate and print some statistics
print("\nEmbedding Statistics:")
print(f"Total examples: {len(texts)}")
print(f"Benign examples: {len(benign_examples)}")
print(f"Non-benign examples: {len(non_benign_examples)}")

# Save a sample of examples to a text file
with open('embedding_examples.txt', 'w') as f:
    f.write("Sample of Benign Examples:\n")
    f.write("="*50 + "\n\n")
    for i, text in enumerate(benign_examples[:10]):  # First 10 benign examples
        f.write(f"{i+1}. {text}\n\n")
    
    f.write("\nSample of Non-Benign Examples:\n")
    f.write("="*50 + "\n\n")
    for i, text in enumerate(non_benign_examples[:10]):  # First 10 non-benign examples
        f.write(f"{i+1}. {text}\n\n")

print("Saved sample examples to 'embedding_examples.txt'")