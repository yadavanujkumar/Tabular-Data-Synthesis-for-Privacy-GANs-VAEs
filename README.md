# Tabular Data Synthesis for Privacy: GANs & VAEs

A Python toolkit for generating statistically realistic, privacy-preserving synthetic tabular data using Generative Adversarial Networks (GANs).

## Overview

This repository demonstrates the creation of synthetic tabular data using Conditional Tabular GAN (CTGAN). The synthetic data retains the statistical utility of the original dataset for training ML models while containing no real Personally Identifiable Information (PII).

## Features

- **Simulated Financial Dataset**: Generates realistic financial data with mixed types (numerical and categorical)
- **CTGAN Training**: Implements the CTGAN training process with categorical column handling
- **Validation Function**: Compares correlation matrices between real and synthetic data
- **Statistical Comparison**: Provides summary statistics and categorical distribution analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the synthetic data generation script:

```bash
python synthetic_data_generation.py
```

The script will:
1. Generate a simulated financial dataset (1000 samples by default)
2. Train a CTGAN model on the data
3. Generate synthetic data of the same size
4. Validate the synthetic data by comparing correlation matrices

## Dataset Structure

The simulated financial dataset includes:

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | String | Unique transaction identifier |
| `customer_age` | Integer | Customer age (18-80) |
| `income` | Integer | Annual income (correlated with age) |
| `transaction_amount` | Float | Transaction amount (correlated with income) |
| `account_type` | Categorical | Checking, Savings, Credit, Investment |
| `transaction_category` | Categorical | Groceries, Entertainment, Bills, Shopping, Transfer |

## Validation Metrics

The validation function compares:
- **Correlation matrices** between real and synthetic data
- **Statistical summaries** (mean, std, min, max, quartiles)
- **Categorical distributions** for discrete columns

A lower mean correlation difference indicates better preservation of statistical relationships.

## Configuration

Key parameters can be adjusted in the script:

```python
N_SAMPLES = 1000    # Number of samples to generate
EPOCHS = 100        # Training epochs (increase for better results)
RANDOM_SEED = 42    # Random seed for reproducibility
```

## Privacy Note

The synthetic data contains NO real PII. All generated data is artificial and suitable for model training, sharing, and analysis without privacy concerns.

## Requirements

- pandas >= 2.0.0
- numpy >= 1.24.0
- ctgan >= 0.10.0

## License

MIT License