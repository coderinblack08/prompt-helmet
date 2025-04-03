# Prompt Injection Protection Demo

This is an interactive demo that showcases how embedding-based classifiers can protect against prompt injection attacks. The demo uses Flask for the backend, a pre-trained embedding model for classification, and Ollama with the Qwen 2.5 (1.5B) model for generating responses.

## Features

- Toggle to enable/disable the embedding classifier protection
- Interactive UI to test prompt injection attempts
- Real-time analysis of potential harmful prompts
- Example injection attempts to try

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed with the Qwen 2.5 (1.5B) model
- Access to the pre-trained embedding model at `../saved_models/embeddings_model_all-MiniLM-L6-v2.bin`

## Setup

1. Make sure you have Ollama installed and the Qwen 2.5 (1.5B) model available:

```bash
ollama list
```

If you don't see `qwen2.5:1.5b` in the list, pull it:

```bash
ollama pull qwen2.5:1.5b
```

2. Ensure the pre-trained embedding model is available at the correct path:

   - The model should be at `../saved_models/embeddings_model_all-MiniLM-L6-v2.bin` relative to the prompt-helmet-demo directory

3. Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Test that the embedding model is available and working:

```bash
python test_model.py
```

5. Start the Flask application:

```bash
python app.py
```

6. Open your browser and navigate to `http://127.0.0.1:5000`

## How It Works

1. The system has a secret (the password "purple unicorn") that it's instructed not to reveal.
2. When the embedding classifier is enabled, it uses a pre-trained sentence transformer model to compare the semantic similarity between the user's prompt and known harmful prompts.
3. If a potential injection is detected, the request is blocked.
4. When the classifier is disabled, the model responds directly to all prompts.

## Try to "Hack" the System

- Try different prompt injection techniques to extract the secret password
- Toggle the classifier on/off to see the difference in protection
- Use the example buttons to test common injection patterns

## Technical Details

- The embedding classifier uses the `all-MiniLM-L6-v2` model from the sentence-transformers library
- Cosine similarity is used to measure the similarity between prompts
- The threshold for detection is set to 0.8 (80% similarity)
- The system uses Ollama's API to generate chat responses with the Qwen 2.5 (1.5B) model

## Troubleshooting

If you encounter issues with the embedding model:

1. Make sure the model file exists at the correct path (`../saved_models/embeddings_model_all-MiniLM-L6-v2.bin`)
2. Run the test script to verify the model is loading correctly:
   ```bash
   python test_model.py
   ```
3. Check that you have all the required dependencies installed
4. Ensure you have enough memory to load the embedding model
