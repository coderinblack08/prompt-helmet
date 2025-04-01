import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
from typing import List, Tuple, Dict, Optional
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disable gradients to save memory
torch.set_grad_enabled(False)

def format_chat_prompt(messages: List[Dict[str, str]], model) -> str:
    """Format a list of messages into a chat prompt that the model can process"""
    
    # If the model's tokenizer has a chat template (like Qwen), use it
    if hasattr(model.tokenizer, 'apply_chat_template'):
        prompt = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    else:
        # Fallback to a simple format
        formatted_prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
        
        formatted_prompt += "Assistant:"
        return formatted_prompt

def generate_text(model, tokens, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Generate text using a manual token-by-token approach"""
    input_tokens = tokens.clone()
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # Forward pass
        with torch.no_grad():
            logits = model(input_tokens)
        
        # Get next token logits
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        # Append to generated tokens
        generated_tokens.append(next_token.item())
        
        # Update input_ids for next iteration
        next_token = next_token.unsqueeze(0)
        input_tokens = torch.cat([input_tokens, next_token], dim=1)
        
        # Break if we hit the end of sequence token
        if next_token.item() == model.tokenizer.eos_token_id:
            break
    
    # Prepare the full output sequence
    full_output = torch.cat([tokens[0], torch.tensor(generated_tokens, device=tokens.device)])
    output_text = model.tokenizer.decode(full_output)
    
    return output_text, generated_tokens

def analyze_head_importance(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    """Measure the importance of each attention head by ablating it and observing loss change"""
    
    # Original loss
    original_loss = model(tokens, return_type="loss")
    
    # Create a tensor to store loss differences
    head_importance = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
    
    # Define a query head ablation hook
    def q_head_ablation_hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint,
        head_idx: int
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        value[:, :, head_idx, :] = 0.
        return value
    
    # Define a key/value head ablation hook for GQA models
    def kv_head_ablation_hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint,
        head_idx: int
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # For GQA models, a single KV head serves multiple query heads
        # Map the query head index to KV head index
        if model.cfg.n_key_value_heads < model.cfg.n_heads:
            # Calculate which KV head corresponds to this query head
            kv_head_idx = head_idx * model.cfg.n_key_value_heads // model.cfg.n_heads
            if kv_head_idx >= model.cfg.n_key_value_heads:
                kv_head_idx = model.cfg.n_key_value_heads - 1
            value[:, :, kv_head_idx, :] = 0.
        else:
            # If it's not GQA, just zero out the corresponding head
            value[:, :, head_idx, :] = 0.
        return value
    
    # Iterate through all heads
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            # Create hook functions for this specific head
            q_hook_fn = lambda value, hook, head_idx=head: q_head_ablation_hook(value, hook, head_idx)
            kv_hook_fn = lambda value, hook, head_idx=head: kv_head_ablation_hook(value, hook, head_idx)
            
            # Run with the hooks to ablate this head
            ablated_loss = model.run_with_hooks(
                tokens,
                return_type="loss",
                fwd_hooks=[
                    (utils.get_act_name("q", layer), q_hook_fn),
                    (utils.get_act_name("k", layer), kv_hook_fn),
                    (utils.get_act_name("v", layer), kv_hook_fn)
                ]
            )
            
            # Store loss difference
            head_importance[layer, head] = ablated_loss.item() - original_loss.item()
    
    return head_importance

def visualize_attention(model: HookedTransformer, tokens: torch.Tensor, layer: int, head: int):
    """Visualize the attention pattern for a specific layer and head"""
    
    # Run model with cache
    _, cache = model.run_with_cache(tokens)
    
    # Get attention pattern
    attention_pattern = cache["pattern", layer][:, head]  # [batch, head, seq_len, seq_len]
    
    # Get token strings
    token_strs = model.to_str_tokens(tokens[0])
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_pattern[0].cpu(), cmap="viridis")
    plt.colorbar()
    plt.title(f"Layer {layer}, Head {head} Attention")
    plt.xticks(range(len(token_strs)), token_strs, rotation=90)
    plt.yticks(range(len(token_strs)), token_strs)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"layer{layer}_head{head}_attention.png")
    plt.close()
    
    print(f"Attention visualization saved to layer{layer}_head{head}_attention.png")

def patch_head_with_custom_pattern(
    model: HookedTransformer, 
    tokens: torch.Tensor, 
    layer: int, 
    head: int,
    custom_pattern: Optional[torch.Tensor] = None
) -> str:
    """Patch a specific attention head with a custom pattern and observe the output"""
    
    # The patching function needs to generate a pattern dynamically
    # based on the current sequence length
    def pattern_patch_hook(
        attn_pattern: Float[torch.Tensor, "batch head_index dest_pos source_pos"],
        hook: HookPoint
    ) -> Float[torch.Tensor, "batch head_index dest_pos source_pos"]:
        # Get the current sequence length
        seq_len = attn_pattern.shape[-1]
        
        # Create a custom pattern for this sequence length
        custom_pattern = torch.zeros((seq_len, seq_len), device=attn_pattern.device)
        
        # Make it attend only to the most recent token
        for i in range(seq_len):
            if i > 0:
                custom_pattern[i, i-1] = 1.0
        
        # Apply the custom pattern to the specified head
        attn_pattern[:, head] = custom_pattern.unsqueeze(0)
        return attn_pattern
    
    # Run the model with the patched attention
    patched_tokens = tokens.clone()
    generated_tokens = []
    
    for i in range(50):  # Generate 50 tokens max
        # Forward pass with hooks
        logits = model.run_with_hooks(
            patched_tokens, 
            fwd_hooks=[(utils.get_act_name("pattern", layer), pattern_patch_hook)]
        )
        
        # Get next token (greedy)
        next_token = logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        
        # Add to generated tokens
        generated_tokens.append(next_token.item())
        
        # Update input for next iteration
        patched_tokens = torch.cat([patched_tokens, next_token], dim=1)
        
        # Break if we hit the end of sequence token
        if next_token.item() == model.tokenizer.eos_token_id:
            break
    
    # Decode the output
    all_tokens = torch.cat([tokens[0], torch.tensor(generated_tokens, device=tokens.device)])
    decoded_output = model.tokenizer.decode(all_tokens)
    
    return decoded_output

def main():
    print(f"Using device: {device}")
    
    # Load the Qwen2.5-0.5B-Instruct model using HookedTransformer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_name}")
    
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device
    )
    
    print(f"Model loaded successfully!")
    print(f"Model config:\n{model.cfg}")
    print(f"Number of query heads: {model.cfg.n_heads}")
    print(f"Number of key/value heads: {model.cfg.n_key_value_heads}")
    
    # Create a chat prompt
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain how transformers work in three sentences."}
    ]
    
    # Format the prompt for the model
    prompt = format_chat_prompt(messages, model)
    print(f"\nFormatted prompt:\n{prompt}")
    
    # Convert prompt to tokens
    tokens = model.to_tokens(prompt)
    print(f"Tokenized prompt shape: {tokens.shape}")
    
    # Generate output 
    print("\nGenerating output:")
    generated_text, _ = generate_text(model, tokens, max_new_tokens=100)
    print(generated_text)
    
    # Run the model with caching to capture internal activations
    logits, cache = model.run_with_cache(tokens)
    
    # Analyze attention patterns
    print("\nAnalyzing attention patterns in the first layer:")
    attention_pattern = cache["pattern", 0]  # Layer 0 attention patterns
    print(f"Attention pattern shape: {attention_pattern.shape}")
    
    # Visualize an attention head
    layer_to_visualize = 0
    head_to_visualize = 0
    print(f"\nVisualizing attention pattern for layer {layer_to_visualize}, head {head_to_visualize}")
    visualize_attention(model, tokens, layer_to_visualize, head_to_visualize)
    
    # Get a subset of heads for the importance analysis
    print("\nAnalyzing head importance (subset to save time):")
    # We'll analyze a subset of layers/heads to save time
    layers_to_analyze = 2
    
    head_importance = torch.zeros((layers_to_analyze, model.cfg.n_heads), device=model.cfg.device)
    
    original_loss = model(tokens, return_type="loss")
    print(f"Original loss: {original_loss.item():.4f}")
    
    # Iterate through a subset of layers
    for layer in range(layers_to_analyze):
        for head in range(model.cfg.n_heads):
            # For GQA, we need to map the query head to the corresponding KV head
            kv_head_idx = head * model.cfg.n_key_value_heads // model.cfg.n_heads
            if kv_head_idx >= model.cfg.n_key_value_heads:
                kv_head_idx = model.cfg.n_key_value_heads - 1
                
            print(f"Analyzing layer {layer}, query head {head}, kv head {kv_head_idx}")
            
            # Define hooks for zeroing out this head
            def q_hook_fn(value, hook):
                value[:, :, head, :] = 0.
                return value
            
            def kv_hook_fn(value, hook):
                value[:, :, kv_head_idx, :] = 0.
                return value
            
            # Run with the hooks to ablate this head
            ablated_loss = model.run_with_hooks(
                tokens,
                return_type="loss",
                fwd_hooks=[
                    (utils.get_act_name("q", layer), q_hook_fn),
                    (utils.get_act_name("k", layer), kv_hook_fn),
                    (utils.get_act_name("v", layer), kv_hook_fn)
                ]
            )
            
            # Store loss difference
            head_importance[layer, head] = ablated_loss.item() - original_loss.item()
    
    # Find the most important head in our subset
    max_importance = head_importance.max()
    max_layer, max_head = torch.where(head_importance == max_importance)
    max_layer, max_head = max_layer[0].item(), max_head[0].item()
    
    print(f"Most important head in subset: Layer {max_layer}, Head {max_head} (Loss change: {max_importance:.4f})")
    
    # Patch a head with a custom pattern
    layer_to_patch = max_layer
    head_to_patch = max_head
    
    print(f"\nPatching layer {layer_to_patch}, head {head_to_patch} with a custom pattern:")
    patched_output = patch_head_with_custom_pattern(model, tokens, layer_to_patch, head_to_patch)
    print(f"Output with patched attention:\n{patched_output}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
