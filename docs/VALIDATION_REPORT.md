# Model Validation Report — Chest X-ray Multi-Label Classifier

Generated: 2026-06-22  |  Model: densenet121  |  Dataset: ChestMNIST (NIH ChestX-ray14)  |  Test n = 22433

All numbers below are produced by `src/evaluate.py` on the held-out test split.

**Mean ROC-AUC across 14 pathologies: 0.8323**

## Per-pathology performance (threshold = 0.5)

| Pathology | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|
| Atelectasis | 0.804 | 0.713 | 0.742 |
| Cardiomegaly | 0.908 | 0.854 | 0.804 |
| Effusion | 0.879 | 0.840 | 0.768 |
| Infiltration | 0.701 | 0.646 | 0.635 |
| Mass | 0.830 | 0.664 | 0.836 |
| Nodule | 0.748 | 0.641 | 0.718 |
| Pneumonia | 0.747 | 0.550 | 0.816 |
| Pneumothorax | 0.871 | 0.690 | 0.889 |
| Consolidation | 0.803 | 0.752 | 0.714 |
| Edema | 0.890 | 0.772 | 0.841 |
| Emphysema | 0.926 | 0.768 | 0.914 |
| Fibrosis | 0.825 | 0.522 | 0.893 |
| Pleural_Thickening | 0.785 | 0.624 | 0.800 |
| Hernia | 0.935 | 0.619 | 0.993 |

## Intended use
Research and educational demonstration of a multi-label chest radiograph classifier with explicit class-imbalance handling and per-class validation. Not a medical device and not for clinical use.

## Limitations
- ChestMNIST is a standardized, down-sampled version of NIH ChestX-ray14; absolute AUCs are not directly comparable to full-resolution CheXNet results.
- Labels are NLP-mined from radiology reports and carry known label noise.
- No external/site-shift validation; performance may degrade on other scanners.

See `docs/PCCP.md` for an illustrative change-control plan for this model.
