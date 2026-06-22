# Predetermined Change Control Plan (PCCP) — Illustrative Example

> **Educational example, not a regulatory submission.** This document shows how
> a Predetermined Change Control Plan (per FDA's Dec 2024 final guidance on
> marketing submissions for AI-enabled device software functions) would be
> structured for a model like this one. It is written to demonstrate the
> author's familiarity with PCCP authoring, not to claim FDA clearance.

## 1. Device / model description
A multi-label deep-learning classifier (DenseNet-121 backbone) that flags 14
thoracic findings on frontal chest radiographs. Outputs per-finding
probabilities for triage support.

## 2. Description of modifications (what may change without a new submission)
- **Retraining on additional data** from the same intended population and
  modality, to improve performance or reduce subgroup gaps.
- **Threshold re-calibration** per finding to maintain target sensitivity.
- **Backbone updates** within the same input/output specification.

Out of scope (would require a new submission): new findings/labels, new
modalities (e.g., lateral views, CT), or change of intended use.

## 3. Modification protocol
- **Data management**: provenance, inclusion/exclusion criteria, and labeling
  QA documented for every added cohort; train/val/test kept site-disjoint.
- **Re-training**: fixed, version-controlled pipeline; configs and seeds logged
  (here via MLflow).
- **Verification & validation**: each candidate model is evaluated on a frozen,
  representative held-out set using the metrics in `VALIDATION_REPORT.md`
  (per-finding ROC-AUC, sensitivity, specificity) plus subgroup slices.
- **Acceptance criteria**: a new version ships only if it is non-inferior on
  mean AUC and does not regress sensitivity on any finding beyond a
  pre-specified margin.

## 4. Impact assessment
Each release is documented with: dataset diff, metric deltas vs the deployed
model, subgroup analysis, and a rollback plan. Drift monitoring (see the
`model-monitoring` project) triggers re-evaluation when input distribution
shifts beyond threshold.

## 5. Transparency
A model card and this PCCP accompany each version; changes are logged and
versioned so reviewers can see exactly what changed and why.
