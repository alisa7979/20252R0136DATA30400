# Hierarchical Multi-Label Text Classification (Zero-Shot)
### DATA304: Big Data Analysis - Final Project

This repository contains the implementation for the DATA304 Final Project. The task is to perform Zero-Shot Hierarchical Multi-Label Text Classification on Amazon product reviews, categorizing them into a taxonomy of 531 classes without explicit labeled training data.

##  Project Overview

We propose a robust pipeline that combines heuristic silver label generation, Large Language Models (LLMs), and Graph Neural Networks (GNNs) to tackle the lack of supervision and class imbalance.

* Task: Multi-label classification (2-3 labels per review).
* Taxonomy: 531 classes with parent-child relationships.
* Best Score (Macro-F1): 0.12 (achieved by final_code.ipynb).

## Repository Structure

| File | Description |
| :--- | :--- |
| final_code.ipynb | [MAIN] The best-performing model. Uses Basic Regex, Random LLM Sampling, Focal Loss, and Self-Training. Run this to reproduce the Kaggle submission. |
| v1.ipynb | [Experiment 1] Variant using Hierarchy Expansion and Class-Weighted Loss. (F1: ~0.08) |
| v2.ipynb | [Experiment 2] Variant adding Weighted Random Sampling (Over-sampling) to v1. (F1: ~0.09) |
| llm_logs.json | Log file containing inputs/outputs of LLM API calls for verification purposes. |
| classes.txt etc. | *Data files (see Setup below).* |

##  How to Reproduce Results

### 1. Environment Setup
Ensure you have Python installed with the following libraries:
```bash
pip install torch transformers numpy pandas scikit-learn networkx openai tqdm
