# Prompt Helmet

> SOTA anomaly-detection algorithm to defend against prompt injection attacks.

References:
https://shekhargulati.com/2024/08/11/building-a-bulletproof-prompt-injection-detector-using-setfit-with-just-32-examples/

## How does it work?

Prompt helmet uses two layers of defense:

1. **Contextual embedding analysis**.

- Uses `text-embedding-3-small` to embed the prompts.

2. **Attention score analysis**.

- Determines when the "distraction affect" triggers in attention heads.

## What dataset does it train on?

[SPML](https://arxiv.org/abs/2402.11755). It is the most comprehensive, public dataset, with system and user prompts clearly delinated.

https://colab.research.google.com/github/HumanCompatibleAI/tensor-trust-data/blob/main/Using%20the%20Tensor%20Trust%20dataset.ipynb#scrollTo=9aacd7ba-7255-47c5-a855-e79168eb4aba

## How does it compare with other solutions?
