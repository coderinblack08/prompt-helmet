# 🪖 Prompt Helmet

> SOTA anomaly-detection algorithm to defend against prompt injection attacks.

## References

- https://shekhargulati.com/2024/08/11/building-a-bulletproof-prompt-injection-detector-using-setfit-with-just-32-examples/

- https://arxiv.org/abs/2402.11755

## Abstract

Large language models (LLMs) are transforming industries with flexible, non-deterministic capabilities, enabling solutions previously unattainable. However, their rapid integration into real-world applications has introduced severe vulnerabilities. Specifically, the Open Web Application Security Project (OWASP) #1 risk for LLMs are prompt injection (PI) attacks. They exploit the inability of LLMs to distinguish between trusted and malicious instructions, leading to risks like data exfiltration and unauthorized actions. Even more concerning, attackers are leveraging optimization techniques to craft computer-generated, adversarial prompts, enabling the creation of novel, unpredictable attacks. Unfortunately, existing solutions, such as system prompt delimiters and supervised fine-tuning (SFT), are insufficient because of their reliance on learning predefined patterns, limiting their ability to generalize. In addition, theoretical limitations suggest that current alignment strategies cannot entirely eliminate adversarial prompts, as current models inherently struggle to ignore these maliciously embedded instructions. This project addresses these vulnerabilities by...

## Results

Embedding model:

- Try different base models:

  - [ ] all-MiniLM-L6-v2 (22.7M)
  - [ ] all-mpnet-base-v2 (109M)

- [ ] Measure the time per inference (ms)

Attention model:

- Try different classifier models:

  - [ ] Gradient Boosting
  - [ ] Random Forest
  - [ ] CNN

- Try different base LLMs:

  - [ ] Qwen (1.5B)
  - [ ] Gemma (2B)
  - [ ] Mistral (7B)
  - [ ] Llama (8B)

- [ ] Measure the time per inference (ms)

Embedding + Attention model:

- [ ] Choose the best embedding model and best attention model

- [ ] Measure the time per inference (ms)
