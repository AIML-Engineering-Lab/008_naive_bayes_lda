# Post 008 — Naive Bayes & LDA: Generative Classifiers

**AI Engineering Lab Series** | Era 1: Classic Machine Learning

## Overview

This project demonstrates Gaussian Naive Bayes and Linear Discriminant Analysis (LDA) for multi-class classification — two fast, interpretable generative models that learn the probability distribution of each class.

| Concept | Description |
|---|---|
| **Naive Bayes** | Classifies by computing P(class\|features) using Bayes' theorem with independence assumption |
| **Gaussian NB** | Assumes each feature follows a Gaussian (normal) distribution per class |
| **LDA** | Finds linear combinations of features that best separate classes |
| **Prior** | P(class) — how common each class is before seeing any features |
| **Likelihood** | P(features\|class) — how likely the features are given the class |
| **Posterior** | P(class\|features) — the final classification probability |

## Datasets

### Dataset A: Jira Bug Ticket Routing
- **Rows:** 8,000 | **Classes:** 6 (Hardware, Firmware, Software, Performance, Security, Docs)
- **Task:** Route incoming bug tickets to the correct engineering team

### Dataset B: Silicon Defect Type Classification (Post-Silicon Validation)
- **Rows:** 6,000 | **Classes:** 6 defect types + Clean Die
- **Task:** Classify wafer defect type from electrical measurements

## Quick Start

```bash
git clone https://github.com/AIML-Engineering-Lab/008_naive_bayes_lda.git
cd 008_naive_bayes_lda
pip install -r requirements.txt
python src/data_generator.py
jupyter notebook notebooks/
```

*Part of the [AI Engineering Lab](https://github.com/AIML-Engineering-Lab) series.*
