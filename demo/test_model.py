import sys
import os
import torch

# Add the parent directory to the path to import the models module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.embeddings_model import ContrastivePromptEmbeddingTrainer
    
    print("Checking for embedding model...")
    
    # Check if CUDA/MPS is available
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("MPS (Apple Silicon) is available")
    else:
        device = "cpu"
        print("Using CPU for inference")
    
    # Try to load the model
    try:
        model_name = "all-MiniLM-L6-v2"
        
        # Try the first path
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "saved_models",
            f"embeddings_model_{model_name}.bin"
        )
        
        # If not found, try the alternative path
        if not os.path.exists(model_path):
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../saved_models",
                f"embeddings_model_{model_name}.bin"
            )
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please ensure the model file exists at this location.")
            sys.exit(1)
            
        print(f"Loading model from: {model_path}")
        embedding_model = ContrastivePromptEmbeddingTrainer(model_name, load_model=True)
        model = embedding_model.get_model()
        
        # Test the model with a simple embedding
        test_text = "This is a test sentence"
        embedding = model.encode(test_text)
        
        print(f"Successfully loaded the model!")
        print(f"Test embedding shape: {len(embedding)}")
        print("The embedding model is working correctly.")
        
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)
        
except ImportError as e:
    print(f"Error importing the embedding model: {e}")
    print("Please make sure the models module is available in the parent directory.")
    sys.exit(1) 