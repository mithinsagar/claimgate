| System | Precision | Recall | F1 |
|---|---|---|---|
| Prompt gpt-3.5-turbo (published, Niu et al. 2024) | 18.8 | 84.4 | 30.8 |
| Prompt gpt-4-turbo (published, Niu et al. 2024) | 33.2 | 90.6 | 45.6 |
| SelfCheckGPT gpt-3.5-turbo (published, Niu et al. 2024) | 35.0 | 58.0 | 43.7 |
| LMvLM gpt-4-turbo (published, Niu et al. 2024) | 18.7 | 76.9 | 30.1 |
| Fine-tuned Llama-2-13B (trained on RAGTruth) (published, Niu et al. 2024) | 61.6 | 76.3 | 68.2 |
| ClaimGate trained on DelucionQA only, threshold from DelucionQA dev | 27.8 | 91.9 | 42.7 (37.9 to 47.3) |
| ClaimGate trained on DelucionQA only, threshold from 300 RAGTruth train responses | 31.2 | 83.8 | 45.5 (40.4 to 50.5) |

ROC AUC 0.768 on 900 test responses.
