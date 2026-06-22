# Chest X-ray Multi Label Classification

A classifier for 14 thoracic findings on chest radiographs. It deals with the
heavy class imbalance you always run into on this dataset, reports AUC,
sensitivity and specificity for every finding instead of a single accuracy
number, draws Grad-CAM overlays so you can see what the model is looking at, and
ships with a change control plan written the way an FDA submission expects one.

Everything runs on NIH ChestX-ray14 through MedMNIST's ChestMNIST, which
downloads on its own with no access request, so the whole pipeline reproduces on
a single GPU.

## Running it

```
pip install -r requirements.txt
python src/train.py --config configs/default.yaml
python src/evaluate.py --checkpoint outputs/best.pt
python src/gradcam.py --checkpoint outputs/best.pt --index 0 --label Cardiomegaly
```

If you are on an NVIDIA card, install the CUDA build of PyTorch that matches it
(cu128 for the 50 series) from pytorch.org first.

## Results

A 10 epoch run on an RTX 5070 Ti, evaluated on the full 22,433 image ChestMNIST
test split, reached a mean ROC-AUC of 0.832 across the 14 findings. Strongest
classes were Hernia 0.935, Emphysema 0.926 and Cardiomegaly 0.908; the hardest
was Infiltration 0.701. The full per class AUC with sensitivity and specificity
sits in `docs/VALIDATION_REPORT.md`.

Those numbers are written by the run itself. Training saves
`outputs/metrics.json` and regenerates the report from the held out split, so
nothing in the repo is hand entered or can go stale. Run it and the report
writes itself.

## What is in here

`dataset.py` loads ChestMNIST and computes the positive weight per class for the
imbalance. `model.py` is the DenseNet121 backbone (the CheXNet baseline) with a
multi label head. `train.py` is the training loop with mixed precision, weighted
BCE, a cosine schedule and MLflow logging. `evaluate.py` produces the per class
metrics and the report. `gradcam.py` saves the attention overlays.
`docs/PCCP.md` is an example change control plan.

## A few honest caveats

ChestMNIST is a standardized, reduced resolution version of ChestX-ray14, so the
AUCs sit a little below what full resolution training gives you. The labels were
mined from radiology reports with NLP and carry that noise. This is a research
and learning project, not a medical device.
