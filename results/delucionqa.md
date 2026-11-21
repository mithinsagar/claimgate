| System | Test macro F1 (671) | 95% CI | Complete contexts (630) | Hallucinated P / R | Test AUC | Dev macro F1 |
|---|---|---|---|---|---|---|
| Sim-overlap (published, Sadat et al. 2023) | 71.09 | | | | | |
| Sim-hybrid (published, Sadat et al. 2023) | 70.81 | | | | | |
| Sim-cosine (published, Sadat et al. 2023) | 69.45 | | | | | |
| Keyword-match (published, Sadat et al. 2023) | 52.77 | | | | | |
| Sim-overlap (Bosch method, our run) | 71.00 | 67.4 to 74.4 | 71.37 | 59.8 / 75.0 |  |  |
| NLI zero-shot (DeBERTa-v3-large) | 75.54 | 72.0 to 79.0 | 75.69 | 79.3 / 57.9 | 0.838 | 78.57 |
| NLI zero-shot + quantity checks | 75.40 | 71.9 to 78.8 | 75.54 | 78.9 / 57.9 | 0.836 | 78.57 |
| LLM judge zero-shot (Qwen2.5-7B-Instruct, 4-bit) | 73.09 | 69.5 to 76.6 | 73.07 | 72.1 / 58.3 | 0.811 | 76.55 |
| ClaimGate | 76.41 | 73.0 to 79.6 | 76.22 | 68.4 / 74.6 | 0.853 | 82.67 |
| ClaimGate without contradiction gate | 76.69 | 73.3 to 79.9 | 76.51 | 68.9 / 74.6 | 0.859 | 82.67 |
| ClaimGate without NLI | 76.93 | 73.5 to 80.2 | 76.38 | 78.8 / 61.9 | 0.850 | 81.86 |
| ClaimGate without quantity checks | 76.10 | 72.7 to 79.4 | 75.88 | 68.0 / 74.2 | 0.852 | 82.19 |
| ClaimGate + LLM judge as a feature | 76.02 | 72.6 to 79.2 | 75.96 | 67.5 / 75.0 | 0.851 | 82.26 |
