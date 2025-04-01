# TransformerLens Analysis with Qwen2.5

This project demonstrates how to use the TransformerLens library to analyze and interpret the Qwen2.5-0.5B-Instruct model.

## Features

- Load Qwen2.5-0.5B-Instruct with TransformerLens
- Format chat prompts using the model's chat template
- Generate completions from the model
- Analyze and visualize attention patterns
- Measure head importance through ablation
- Patch attention heads with custom patterns
- Support for Grouped Query Attention (GQA) architecture

## Qwen2.5 Architecture

The Qwen2.5 models use Grouped Query Attention (GQA), where:

- The number of query heads (14) is larger than the number of key/value heads (2)
- Multiple query heads share the same key/value head
- This requires special handling when ablating or analyzing attention heads

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the analysis script:

```bash
python qwen_analysis.py
```

## What Happens

The script:

1. Loads the Qwen2.5-0.5B-Instruct model using TransformerLens
2. Creates a formatted chat prompt
3. Generates a completion from the model
4. Analyzes attention patterns in the first layer
5. Visualizes an attention head and saves it as an image
6. Analyzes the importance of each attention head, accounting for GQA
7. Identifies the most important head
8. Patches that head with a custom attention pattern and observes the output

## Functions

- `format_chat_prompt()`: Formats chat messages for the model
- `generate_text()`: Custom text generation function with sampling
- `analyze_head_importance()`: Measures importance of attention heads
- `visualize_attention()`: Creates visualizations of attention patterns
- `patch_head_with_custom_pattern()`: Modifies attention patterns

## GQA Handling

When working with GQA models like Qwen2.5, the script:

1. Maps query head indices to key/value head indices
2. Correctly ablates both the query and corresponding key/value heads
3. Generates dynamic attention patterns that adapt to changing sequence lengths

## Modifying the Code

To change the model or prompt:

1. Update the `model_name` variable to use a different model
2. Modify the `messages` list to change the prompt

To perform different analyses:

1. Use the `run_with_cache()` method to access model activations
2. Create custom hooks with `run_with_hooks()` to modify model behavior

## Advanced Usage

For more advanced analyses, you can:

- Implement different patching strategies
- Create hooks to capture and modify any intermediate activations
- Analyze specific circuits within the model
