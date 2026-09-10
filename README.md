# Antibody Quality Attribute Machine Learning Pipeline

## Overview
This repository provides a robust, high-integrity machine learning pipeline designed to predict antibody developability, biophysical stability, and quality attributes (e.g., $HMW\%$, Polyreactivity, $K_d$) directly from sequence and structural data. 

To overcome the inherent biases of biological datasets, the pipeline implements exhaustive combinatorial grid searches across structural subregions and feature groups while strictly enforcing mathematical boundaries against **data leakage**, **multicollinearity**, and **combinatorial explosion**.

---

## Architecture & Feature Engineering

The pipeline systematically translates raw amino acid sequences and 3D structures into quantitative feature matrices before executing cross-validated model selection.

### 1. Structural Regions
Sequences are parsed and stitched into distinct functional domains for evaluation:
* **`CD3_VH`**: The complete Heavy Chain variable domain.
* **`CD3_VL`**: The complete Light Chain variable domain.
* **`scFv`**: The single-chain variable fragment, constructed by linking the VH and VL domains with their native spacer sequence.

### 2. Feature Families
Features are segregated into isolated groups to prevent redundant information from diluting tree-based splitting or destabilizing linear models:
* **AAC**: Amino Acid Composition (raw frequency of the 20 standard amino acids).
* **AAindex**: Over 566 raw biochemical and biophysical properties mapped from the literature.
* **Georgiev**: 19-dimensional Principal Component Analysis (PCA) compression of amino acid properties.
* **Protein Language Models (ESM-2 / AntiBERTy / AbLang2)**: Transformer-based neural embeddings capturing evolutionary and structural context across varying parameter scales (8M, 35M, 650M).
* **Propermab**: 3D structural physics (surface charge, electrostatics, hydrophobicity) generated via automated antibody structure modeling (ABodyBuilder2).
* **CQA / Custom Tabular**: Direct assay readouts or experimental metadata (e.g., media type, expression titer) injected into the feature space.

### 3. Global vs. Targeted Pooling
* **Global Pooling**: Features are averaged across an entire structural domain.
* **Targeted Pooling**: Features are restricted strictly to user-defined spatial hotspots (e.g., Chothia interface residues). The pipeline uses cryptographic MD5 hashing on the residue index lists to generate unique, version-controlled cache files (`.npz`), ensuring targeted and global feature spaces never collide.

---

## Validation & Cross-Validation Protocol

To guarantee that models generalize reliably to unseen antibody variants rather than memorizing sequence scaffolds, the validation architecture integrates strict statistical defenses:

* **Repeated Stratified K-Fold / Out-of-Fold Cross-Validation**: Evaluates feature combinations using robust out-of-fold predictions to prevent optimistic bias.
* **Out-of-Group (OOG) Generalization Testing**: Supports custom split groups (e.g., variant generations or experimental batches) to explicitly test model transferability from training lineages to entirely unseen variant classes.
* **Ranking & Enrichment Metrics**: Models are primarily optimized and ranked using **Spearman's Rank Correlation** ($r_s$) alongside $R^2$, RMSE, and MAE, backed by automated enrichment curves to measure the recovery of top-performing antibody candidates.

---

## Configuration Guide (`main` block)

All pipeline runs are fully controlled via parameters in the `main()` function, requiring zero modifications to core script logic.

| Variable | Type | Description |
| :--- | :--- | :--- |
| `filepath` | String | Path to the input dataset CSV file. |
| `targets_to_test` | Dictionary | Target column names mapped to classification cutoffs (e.g., `{'HMW%': 10.0}`). |
| `models_to_test` | List | Algorithms to evaluate (e.g., `['SVR', 'PLSRegression', 'XGBoost', 'RandomForest', 'ElasticNet']`). |
| `ENABLE_TARGETED_FEATURES` | Boolean | Master switch to toggle targeted residue pooling. |
| `my_target_indices` | Dictionary | Maps region names to a semantic description tag and a list of 0-indexed integer positions. |
| `FEATURES_TO_EXCLUDE` | List | Blacklist of feature groups to block from the grid search (e.g., to prevent SVD data leakage). |
| `csv_feature_columns` | List | Tabular metadata columns in the CSV to merge into the combinatorial feature space. |
| `RUN_SINGLE_EVAL` | Boolean | Bypasses the exhaustive search to test a single predefined region and feature combination for debugging. |
| `GENERATE_EXTENDED_PLOTS` | Boolean | Toggles the generation of comprehensive 3x3 diagnostic and learning curve dashboards. |

---

## Exhaustive Search Safety Constraints

The combinatorial grid search automatically enforces structural and statistical rules before evaluating any combination:
1. **Mutually Exclusive Properties**: `AAindex` and `Georgiev` cannot occupy the same model to avoid feature redundancy.
2. **Global/Target Isolation**: Global and Targeted variants of identical feature families are barred from mixing.
3. **Neural Network Limits**: Restricted to a maximum of one deep language model per model run to prevent memory overflow and overfitting.
4. **Paired Structural Rules**: Complex 3D structural models (`Propermab`, `AbLang2`) require a fully paired context (`CD3_VH` + `CD3_VL` or `scFv`) to activate.
