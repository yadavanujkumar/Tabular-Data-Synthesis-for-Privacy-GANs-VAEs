#!/usr/bin/env python3
"""
Synthetic Tabular Data Generation using CTGAN

This script demonstrates the creation of statistically realistic, but completely
synthetic, tabular data using a Conditional Tabular GAN (CTGAN).

Goal: Generate a synthetic dataset that retains the statistical utility of the
original for training ML models but contains no real PII.

Author: Data Privacy Scientist
"""

import numpy as np
import pandas as pd
from ctgan import CTGAN


def generate_simulated_financial_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a simulated financial dataset with mixed data types.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Simulated financial dataset
    """
    np.random.seed(seed)

    # Generate transaction IDs
    transaction_ids = [f"TXN_{i:06d}" for i in range(1, n_samples + 1)]

    # Generate customer ages (18-80)
    customer_ages = np.random.randint(18, 81, size=n_samples)

    # Generate income with some correlation to age
    # Income increases with age until around 50, then plateaus
    age_factor = np.minimum(customer_ages, 50) - 18  # Cap at age 50
    base_income = 30000 + age_factor * 1500
    income_noise = np.random.normal(0, 10000, size=n_samples)
    incomes = np.maximum(20000, base_income + income_noise).astype(int)

    # Generate transaction amounts correlated with income
    # Higher income tends to mean higher transaction amounts
    transaction_amounts = (incomes * np.random.uniform(0.01, 0.05, size=n_samples)).round(2)

    # Generate categorical columns
    account_types = np.random.choice(
        ["Checking", "Savings", "Credit", "Investment"],
        size=n_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    )

    transaction_categories = np.random.choice(
        ["Groceries", "Entertainment", "Bills", "Shopping", "Transfer"],
        size=n_samples,
        p=[0.3, 0.15, 0.25, 0.2, 0.1]
    )

    # Create DataFrame
    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "customer_age": customer_ages,
        "income": incomes,
        "transaction_amount": transaction_amounts,
        "account_type": account_types,
        "transaction_category": transaction_categories
    })

    return df


def train_ctgan_model(
    real_data: pd.DataFrame,
    discrete_columns: list,
    epochs: int = 300,
    verbose: bool = True
) -> CTGAN:
    """
    Train a CTGAN model on the real data.

    Parameters
    ----------
    real_data : pd.DataFrame
        The real dataset to train on
    discrete_columns : list
        List of column names that are categorical/discrete
    epochs : int
        Number of training epochs
    verbose : bool
        Whether to print training progress

    Returns
    -------
    CTGAN
        Trained CTGAN model
    """
    # Initialize CTGAN with reasonable parameters for tabular data
    ctgan = CTGAN(
        epochs=epochs,
        verbose=verbose
    )

    # Define the columns to exclude from training (identifiers)
    # We'll drop transaction_id as it's a unique identifier
    training_data = real_data.drop(columns=["transaction_id"])

    # Update discrete columns to exclude transaction_id if present
    training_discrete_cols = [col for col in discrete_columns if col != "transaction_id"]

    # Fit the model
    print("\n" + "=" * 60)
    print("Training CTGAN Model...")
    print("=" * 60)
    print(f"Training samples: {len(training_data)}")
    print(f"Features: {list(training_data.columns)}")
    print(f"Categorical columns: {training_discrete_cols}")
    print("=" * 60 + "\n")

    ctgan.fit(training_data, training_discrete_cols)

    return ctgan


def generate_synthetic_data(
    ctgan_model: CTGAN,
    n_samples: int
) -> pd.DataFrame:
    """
    Generate synthetic data using the trained CTGAN model.

    Parameters
    ----------
    ctgan_model : CTGAN
        Trained CTGAN model
    n_samples : int
        Number of synthetic samples to generate

    Returns
    -------
    pd.DataFrame
        Synthetic dataset
    """
    print("\n" + "=" * 60)
    print(f"Generating {n_samples} synthetic samples...")
    print("=" * 60 + "\n")

    # Generate synthetic data
    synthetic_data = ctgan_model.sample(n_samples)

    # Add synthetic transaction IDs
    synthetic_data.insert(0, "transaction_id", [f"SYN_{i:06d}" for i in range(1, n_samples + 1)])

    return synthetic_data


def compute_correlation_matrix(df: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
    """
    Compute the correlation matrix for numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    numerical_columns : list
        List of numerical column names

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    return df[numerical_columns].corr()


def validate_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    numerical_columns: list,
    categorical_columns: list = None
) -> dict:
    """
    Validate synthetic data by comparing correlation matrices.

    Parameters
    ----------
    real_data : pd.DataFrame
        Original real dataset
    synthetic_data : pd.DataFrame
        Generated synthetic dataset
    numerical_columns : list
        List of numerical column names to compare
    categorical_columns : list, optional
        List of categorical column names to compare distributions

    Returns
    -------
    dict
        Validation results including correlation matrices and comparison
    """
    print("\n" + "=" * 60)
    print("Validating Synthetic Data Quality")
    print("=" * 60)

    # Compute correlation matrices
    real_corr = compute_correlation_matrix(real_data, numerical_columns)
    synthetic_corr = compute_correlation_matrix(synthetic_data, numerical_columns)

    # Compute the difference between correlation matrices
    corr_diff = np.abs(real_corr - synthetic_corr)

    # Handle case with fewer than 2 numerical columns
    upper_tri_indices = np.triu_indices_from(corr_diff.values, k=1)
    if len(upper_tri_indices[0]) > 0:
        mean_corr_diff = corr_diff.values[upper_tri_indices].mean()
    else:
        mean_corr_diff = 0.0  # No correlation pairs to compare

    print("\n--- Real Data Correlation Matrix ---")
    print(real_corr.round(4).to_string())

    print("\n--- Synthetic Data Correlation Matrix ---")
    print(synthetic_corr.round(4).to_string())

    print("\n--- Absolute Difference (Real - Synthetic) ---")
    print(corr_diff.round(4).to_string())

    print(f"\n--- Mean Correlation Difference (Upper Triangle): {mean_corr_diff:.4f} ---")

    # Additional statistical comparison
    print("\n--- Statistical Summary Comparison ---")
    print("\nReal Data Statistics:")
    print(real_data[numerical_columns].describe().round(2).to_string())

    print("\nSynthetic Data Statistics:")
    print(synthetic_data[numerical_columns].describe().round(2).to_string())

    # Categorical distribution comparison
    if categorical_columns:
        print("\n--- Categorical Distribution Comparison ---")
        for col in categorical_columns:
            if col in real_data.columns and col in synthetic_data.columns:
                print(f"\n{col}:")
                real_dist = real_data[col].value_counts(normalize=True).sort_index()
                synthetic_dist = synthetic_data[col].value_counts(normalize=True).sort_index()

                comparison = pd.DataFrame({
                    "Real": real_dist,
                    "Synthetic": synthetic_dist
                }).fillna(0)
                comparison["Difference"] = np.abs(comparison["Real"] - comparison["Synthetic"])
                print(comparison.round(4).to_string())

    # Return validation results
    return {
        "real_correlation": real_corr,
        "synthetic_correlation": synthetic_corr,
        "correlation_difference": corr_diff,
        "mean_correlation_difference": mean_corr_diff
    }


def main():
    """
    Main function to demonstrate synthetic data generation with CTGAN.
    """
    print("=" * 60)
    print("Synthetic Tabular Data Generation using CTGAN")
    print("=" * 60)

    # Configuration
    N_SAMPLES = 1000
    EPOCHS = 100  # Reduced for demonstration; increase for better results
    RANDOM_SEED = 42

    # Define column types
    NUMERICAL_COLUMNS = ["customer_age", "income", "transaction_amount"]
    CATEGORICAL_COLUMNS = ["account_type", "transaction_category"]
    DISCRETE_COLUMNS = CATEGORICAL_COLUMNS  # For CTGAN

    # Step 1: Generate simulated financial data
    print("\n" + "=" * 60)
    print("Step 1: Generating Simulated Financial Data")
    print("=" * 60)

    real_data = generate_simulated_financial_data(n_samples=N_SAMPLES, seed=RANDOM_SEED)

    print(f"\nGenerated {len(real_data)} samples of simulated financial data")
    print("\nSample of real data:")
    print(real_data.head(10).to_string(index=False))
    print(f"\nData types:\n{real_data.dtypes}")

    # Step 2: Train CTGAN model
    print("\n" + "=" * 60)
    print("Step 2: Training CTGAN Model")
    print("=" * 60)

    ctgan_model = train_ctgan_model(
        real_data=real_data,
        discrete_columns=DISCRETE_COLUMNS,
        epochs=EPOCHS,
        verbose=True
    )

    # Step 3: Generate synthetic data
    print("\n" + "=" * 60)
    print("Step 3: Generating Synthetic Data")
    print("=" * 60)

    synthetic_data = generate_synthetic_data(ctgan_model, n_samples=N_SAMPLES)

    print("\nSample of synthetic data:")
    print(synthetic_data.head(10).to_string(index=False))

    # Step 4: Validate synthetic data
    print("\n" + "=" * 60)
    print("Step 4: Validating Synthetic Data")
    print("=" * 60)

    validation_results = validate_synthetic_data(
        real_data=real_data,
        synthetic_data=synthetic_data,
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\n✓ Generated {len(real_data)} real samples")
    print(f"✓ Trained CTGAN model with {EPOCHS} epochs")
    print(f"✓ Generated {len(synthetic_data)} synthetic samples")
    print(f"✓ Mean correlation difference: {validation_results['mean_correlation_difference']:.4f}")

    if validation_results['mean_correlation_difference'] < 0.1:
        print("\n✓ Synthetic data preserves statistical relationships well!")
    elif validation_results['mean_correlation_difference'] < 0.2:
        print("\n⚠ Synthetic data moderately preserves statistical relationships.")
        print("  Consider increasing training epochs for better results.")
    else:
        print("\n⚠ Synthetic data has significant correlation differences.")
        print("  Recommend increasing training epochs or adjusting model parameters.")

    print("\n" + "=" * 60)
    print("Privacy Note: The synthetic data contains NO real PII.")
    print("All generated data is artificial and suitable for model training.")
    print("=" * 60)

    return real_data, synthetic_data, validation_results


if __name__ == "__main__":
    main()
