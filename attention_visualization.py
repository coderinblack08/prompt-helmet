import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

def create_attention_visualization(sentence, attention_matrix=None):
    """
    Create a visualization of attention patterns between words in a sentence.
    
    Args:
        sentence (str): The input sentence to visualize
        attention_matrix (numpy.ndarray, optional): Custom attention matrix.
            If None, a sample attention pattern will be generated.
    """
    # Split the sentence into words
    words = sentence.split()
    n_words = len(words)
    
    # If no attention matrix is provided, create a sample one
    if attention_matrix is None:
        # Create a sample attention matrix (normally this would come from a model)
        attention_matrix = np.random.rand(n_words, n_words)
        # Make diagonal elements stronger (self-attention)
        np.fill_diagonal(attention_matrix, np.random.uniform(0.7, 1.0, n_words))
        # Add some structure - words next to each other tend to have higher attention
        for i in range(n_words):
            for j in range(n_words):
                if abs(i-j) == 1:  # Adjacent words
                    attention_matrix[i, j] = np.random.uniform(0.5, 0.9)
    
    # Normalize the attention matrix
    attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(14, 10), facecolor='black')
    
    # Define custom colors for word boxes based on the example
    word_colors = {
        'a': '#444444',        # gray
        'fluffy': '#1E90FF',   # blue
        'blue': '#1E90FF',     # blue
        'creature': '#444444', # gray
        'roamed': '#444444',   # gray
        'the': '#444444',      # gray
        'verdant': '#228B22',  # green
        'forest': '#444444',   # gray
    }
    
    # Default color for words not in the dictionary
    default_color = '#444444'  # gray
    
    # Create a grid for the visualization - make sure we have enough rows and columns
    gs = gridspec.GridSpec(n_words + 3, n_words + 1, figure=fig)
    
    # Display the sentence at the top
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.text(0.5, 0.5, ' '.join(words), fontsize=16, color='white', 
                 ha='center', va='center')
    ax_title.axis('off')
    
    # Create the top row with word boxes and arrows
    for i, word in enumerate(words):
        # Word box
        ax_word = fig.add_subplot(gs[1, i])
        color = word_colors.get(word.lower(), default_color)
        ax_word.text(0.5, 0.5, word, fontsize=12, color='white',
                    ha='center', va='center', 
                    bbox=dict(facecolor=color, edgecolor='white', boxstyle='square'))
        ax_word.axis('off')
        
        # Arrow down to embedding
        ax_arrow = fig.add_subplot(gs[2, i])
        ax_arrow.arrow(0.5, 0.8, 0, -0.6, head_width=0.1, head_length=0.1, 
                      fc='white', ec='white', width=0.02)
        ax_arrow.text(0.5, 0.2, f"E{i+1}", fontsize=12, color='white', 
                     ha='center', va='center')
        ax_arrow.axis('off')
    
    # Create the attention matrix visualization
    for i in range(n_words):
        for j in range(n_words):
            ax = fig.add_subplot(gs[i+3, j])
            
            # Draw a circle with size proportional to attention weight
            attention_weight = attention_matrix[i, j]
            
            # Only show significant attention (threshold can be adjusted)
            if attention_weight > 0.1:
                # Create a circle with size proportional to attention weight
                circle_size = attention_weight * 0.8  # Scale factor
                circle = Circle((0.5, 0.5), circle_size/2, 
                               facecolor='gray', alpha=0.5, edgecolor='none')
                ax.add_patch(circle)
                
                # Add the attention value text
                if attention_weight > 0.2:  # Only show text for significant attention
                    ax.text(0.5, 0.5, f"K{i+1}·Q{j+1}", fontsize=8, color='#00FFFF',
                           ha='center', va='center')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        # Add row labels (words)
        ax_label = fig.add_subplot(gs[i+3, n_words])
        color = word_colors.get(words[i].lower(), default_color)
        ax_label.text(0.1, 0.5, words[i], fontsize=12, color='white',
                     ha='left', va='center', 
                     bbox=dict(facecolor=color, edgecolor='white', boxstyle='square'))
        
        # Add embedding and key vectors
        ax_label.text(0.5, 0.5, f" → E{i+1} → K{i+1}", fontsize=10, color='#00FFFF',
                     ha='left', va='center')
        ax_label.axis('off')
    
    # Add arrows for high attention relationships
    for i in range(n_words):
        for j in range(n_words):
            # Only draw arrows for high attention values
            if attention_matrix[i, j] > 0.4 and i != j:
                # Find the source and target axes
                source_ax = fig.add_subplot(gs[1, i])
                target_ax = fig.add_subplot(gs[1, j])
                
                # Get the positions in figure coordinates
                source_pos = source_ax.get_position()
                target_pos = target_ax.get_position()
                
                # Create an arrow
                arrow = FancyArrowPatch(
                    (source_pos.x0 + source_pos.width/2, source_pos.y0),
                    (target_pos.x0 + target_pos.width/2, target_pos.y0 - 0.05),
                    transform=fig.transFigure,
                    connectionstyle="arc3,rad=0.2",
                    arrowstyle="simple,head_width=5,head_length=10",
                    color='yellow',
                    alpha=min(1.0, attention_matrix[i, j] * 2),  # Scale opacity with attention
                    linewidth=1.5
                )
                fig.patches.append(arrow)
    
    plt.suptitle("Attention Pattern Visualization", color='white', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("attention_visualization.png", facecolor='black', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Example sentence (from the image)
    sentence = "a fluffy blue creature roamed the verdant forest"
    
    # Create a custom attention matrix that highlights certain relationships
    n_words = len(sentence.split())
    attention_matrix = np.random.rand(n_words, n_words) * 0.2  # Base low attention
    
    # Add specific attention patterns
    # For example: "fluffy" and "creature" have high attention
    word_indices = {word: i for i, word in enumerate(sentence.split())}
    
    # "fluffy" attends to "creature"
    attention_matrix[word_indices["fluffy"], word_indices["creature"]] = 0.8
    
    # "blue" attends to "creature"
    attention_matrix[word_indices["blue"], word_indices["creature"]] = 0.7
    
    # "creature" attends to "roamed"
    attention_matrix[word_indices["creature"], word_indices["roamed"]] = 0.6
    
    # "roamed" attends to "forest"
    attention_matrix[word_indices["roamed"], word_indices["forest"]] = 0.7
    
    # "the" attends to "forest"
    attention_matrix[word_indices["the"], word_indices["forest"]] = 0.8
    
    # "verdant" attends to "forest"
    attention_matrix[word_indices["verdant"], word_indices["forest"]] = 0.9
    
    # Create the visualization
    create_attention_visualization(sentence, attention_matrix)

if __name__ == "__main__":
    main() 