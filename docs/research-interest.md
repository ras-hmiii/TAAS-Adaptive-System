# Research Interest: Traceable and Adaptive AI Systems

## Project overview

TAAS (Traceable and Adaptive AI System) is a research-oriented machine learning
framework for studying model reliability under changing data distributions. It
connects four concerns that are often developed separately:

1. **Prediction** - a model produces a classification decision and confidence.
2. **Explanation** - the system identifies why a prediction was made.
3. **Monitoring** - model behavior and performance are measured over time.
4. **Adaptation** - detected changes trigger a controlled candidate-model
   evaluation rather than an automatic, untraceable replacement.

The central idea is to make adaptation observable and accountable. A detected
drift event is not treated as proof that a model should be replaced. Instead,
TAAS records the event, evaluates current and candidate performance, applies an
explicit decision rule, and preserves an audit trail.

## Research motivation

Machine learning systems are usually evaluated before deployment, while their
inputs and operating conditions continue to change after deployment. A model
can therefore become less reliable even when its implementation has not
changed. This creates several practical and scientific problems:

- A performance decline may be caused by data drift, concept drift, or a change
  in the evaluation population.
- A drift detector may raise false alarms or detect change too late.
- Retraining may improve current performance while causing instability or
  forgetting of previously learned behavior.
- Explanations may change as the model adapts, making model decisions harder to
  compare across versions.

TAAS focuses on the relationship between these problems instead of treating
explainability, monitoring, and learning as isolated features.

## Main research question

> Can explainability, concept-drift detection, and controlled model adaptation
> work together to maintain model reliability when real-world data
> distributions change?

## Supporting research questions

### RQ1 - Explanation reliability

How faithful and stable are SHAP explanations for tabular predictions before
and after a distribution shift?

Relevant measurements include:

- SHAP reconstruction error
- explanation fidelity score
- ranking of the most influential features
- change in feature contributions before and after drift

### RQ2 - Drift detection behavior

How do ADWIN and Page-Hinkley differ when detecting controlled changes in a
stream of model observations?

Relevant measurements include:

- detection delay
- false alarms
- missed detections
- detector sensitivity to shift magnitude
- detector behavior under noise

### RQ3 - Adaptation effectiveness

When drift is detected, does controlled adaptation recover model performance
more effectively than leaving the baseline model unchanged?

Relevant comparisons include:

- no adaptation
- periodic retraining
- incremental learning
- incremental learning with EWC regularization

### RQ4 - Stability and forgetting

Does adaptation improve performance on post-drift data without substantially
damaging performance on pre-drift data?

Relevant measurements include:

- pre-drift performance
- post-drift performance
- recovery performance
- recovery time
- adaptation improvement
- forgetting score

## Current system architecture

```text
Data ingestion
      |
      v
Preprocessing and deterministic split
      |
      v
Baseline model
      |
      +--> Prediction and confidence
      |
      +--> SHAP explanation
      |
      v
Controlled drift simulation
      |
      +--> ADWIN
      +--> Page-Hinkley
      |
      v
Performance monitoring
      |
      v
Adaptation decision
      |
      +--> Candidate model training
      +--> Candidate evaluation
      +--> Explicit promotion decision
      |
      v
Model metadata and audit trail
```

The implementation is intentionally modular. Detection logic is independent of
the dashboard, and the dashboard reads saved artifacts rather than embedding
machine learning logic in UI code.

## Implemented research components

### Tabular baseline

The current baseline uses the UCI Adult Income dataset and a Random Forest
classifier. The ingestion module:

- downloads or loads a local dataset
- handles missing values
- encodes categorical variables
- creates deterministic train, validation, and test splits
- preserves encoders for later inference and explanation

### Explainability

The tabular pipeline uses SHAP TreeExplainer for individual and batch
explanations. The image module provides optional ResNet50/ImageNet inference
and Grad-CAM heatmap overlays when TensorFlow is installed.

### Drift simulation and detection

Experiments can introduce deterministic mean shifts, variance shifts, or noise
into a numeric stream. Detector wrappers expose structured results containing:

- detector name
- batch/index
- observed value
- detection status
- detector score where available
- drift event metadata

### Controlled adaptation

The adaptation workflow distinguishes:

1. drift detected
2. candidate evaluation required
3. candidate recommended for promotion
4. candidate rejected
5. production promotion

This prevents a detector signal from silently replacing the active model.

### Streaming experiments

The streaming runner processes batches, records pre-adaptation metrics, trains
candidate models after drift, calculates post-adaptation metrics, and records
audit events. Candidate promotion remains an explicit decision.

### Traceability

Experiment results and model metadata can be persisted as JSON artifacts.
Structured audit events are available for batch receipt, drift detection, and
adaptation decisions.

## Proposed experimental methodology

1. Train a baseline model on the pre-drift training split.
2. Construct a deterministic stream with a known drift point.
3. Run the stream with ADWIN and Page-Hinkley independently.
4. Measure performance before and after the known drift point.
5. Record detector events and calculate detection delay.
6. Train a candidate using recent post-drift observations.
7. Compare candidate and baseline on both post-drift and retained pre-drift
   evaluation data.
8. Promote only when the candidate satisfies a predefined improvement and
   stability threshold.
9. Repeat across seeds, drift types, and shift magnitudes.
10. Compare the resulting experiment artifacts rather than relying on a single
    run.

## Expected contribution

The intended contribution is not a new classifier. It is an experimental
framework for evaluating the interaction between:

- interpretable predictions
- streaming drift detection
- conservative model adaptation
- model-version traceability

The framework can make a useful distinction between a system that merely
detects change and a system that can justify, validate, and audit its response
to change.

## Limitations

- The current baseline is tabular and uses a public benchmark dataset.
- The model registry is intentionally lightweight and is not a production
  model-serving registry.
- Image functionality currently supports inference and Grad-CAM utilities, not
  a complete domain-specific image training study.
- Large-scale incremental-learning and EWC comparisons still require execution
  on defined experimental streams.
- Research conclusions must not be drawn until repeated experiments produce
  sufficient measurements.

## Recommended next experiment

Run a reproducible Adult Income stream experiment with:

- one fixed pre-drift training split
- a known feature-distribution shift
- ADWIN and Page-Hinkley
- baseline/no adaptation
- periodic retraining
- incremental adaptation
- incremental adaptation with EWC

Report detection delay, false alarms, F1 before drift, F1 after drift, recovery
F1, recovery time, and forgetting score for every configuration and seed.

## Research interest statement

My research interest is in building reliable machine learning systems that can
explain their decisions, recognize when their operating environment changes,
and adapt without losing traceability or control. TAAS provides a practical
experimental setting for investigating this intersection of explainable AI,
concept drift detection, continual learning, and model governance.
