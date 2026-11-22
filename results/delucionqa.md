| System | Test macro F1 | 95% CI | Unseen contexts | Hallucinated P / R | Test AUC | p (ClaimGate not better) |
|---|---|---|---|---|---|---|
| Sim-overlap (published, Sadat et al. 2023) | 71.09 | | | | | |
| Sim-hybrid (published, Sadat et al. 2023) | 70.81 | | | | | |
| Sim-cosine (published, Sadat et al. 2023) | 69.45 | | | | | |
| Keyword-match (published, Sadat et al. 2023) | 52.77 | | | | | |
| Sim-overlap (Bosch method, our run) | 71.14 | 67.6 to 74.6 | 71.78 | 60.0 / 75.0 |  | 0.000 |
| NLI zero-shot (DeBERTa-v3-large) | 75.48 | 72.0 to 78.9 | 72.76 | 79.7 / 57.5 | 0.842 | 0.051 |
| HHEM-2.1-Open zero-shot | 74.97 | 71.5 to 78.3 | 74.81 | 70.0 / 66.7 | 0.844 | 0.022 |
| MiniCheck-Flan-T5-Large zero-shot | 71.27 | 67.7 to 74.7 | 71.45 | 66.5 / 59.9 | 0.795 | 0.000 |
| LLM judge zero-shot (Qwen2.5-7B-Instruct, 4-bit) | 73.09 | 69.5 to 76.6 | 71.74 | 72.1 / 58.3 | 0.811 | 0.005 |
| Fine-tuned verifier alone | 78.25 | 75.0 to 81.4 | 79.12 | 72.7 / 73.0 | 0.861 | 0.565 |
| ClaimGate | 78.00 | 74.6 to 81.2 | 77.03 | 74.1 / 70.2 | 0.855 |  |
| ClaimGate, no fine-tuned network | 78.00 | 74.6 to 81.2 | 77.03 | 74.1 / 70.2 | 0.855 | 1.000 |
| ClaimGate variant: verifier alone (not selected on dev) | 78.32 | 75.1 to 81.4 | 79.17 | 72.4 / 73.8 | 0.859 | 0.585 |
| ClaimGate variant: equal-weight blend (not selected on dev) | 76.31 | 72.8 to 79.7 | 74.53 | 85.4 / 55.6 | 0.867 | 0.119 |
| ClaimGate without contradiction gate | 78.10 | 74.7 to 81.3 | 77.22 | 74.6 / 69.8 | 0.861 | 0.659 |
| ClaimGate without NLI features | 77.50 | 74.2 to 80.7 | 78.54 | 70.6 / 74.2 | 0.848 | 0.317 |
| ClaimGate without quantity features | 78.21 | 74.9 to 81.4 | 77.22 | 75.1 / 69.4 | 0.857 | 0.693 |
| ClaimGate without rule-made contradictions in training | 78.00 | 74.6 to 81.2 | 77.03 | 74.1 / 70.2 | 0.855 | 1.000 |
