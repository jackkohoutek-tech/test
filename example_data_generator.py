#!/usr/bin/env python3
"""
Generate example datasets for testing the XGBoost training application.
This script creates sample CSV files with different characteristics.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

def generate_classification_data(n_samples=1000, n_features=10, 
                                 n_classes=2, output_file='data_classification.csv'):
    """Generate a classification dataset."""
    print(f"Generating classification dataset: {output_file}")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        n_redundant=2,
        n_classes=n_classes,
        random_state=42,
        flip_y=0.1
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"  Saved {n_samples} samples with {n_features} features and {n_classes} classes")
    print(f"  Class distribution: {np.bincount(y)}")


def generate_regression_data(n_samples=1000, n_features=10, 
                             output_file='data_regression.csv'):
    """Generate a regression dataset."""
    print(f"Generating regression dataset: {output_file}")
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        noise=10.0,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"  Saved {n_samples} samples with {n_features} features")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")


def generate_test_files(n_files=3, base_name='test', task='classification'):
    """Generate multiple test files."""
    print(f"\nGenerating {n_files} test files for {task}...")
    
    for i in range(n_files):
        output_file = f'{base_name}_{i+1}.csv'
        
        if task == 'classification':
            generate_classification_data(
                n_samples=200,
                n_features=10,
                n_classes=2,
                output_file=output_file
            )
        else:
            generate_regression_data(
                n_samples=200,
                n_features=10,
                output_file=output_file
            )


if __name__ == '__main__':
    print("="*70)
    print("XGBoost Training App - Example Data Generator")
    print("="*70)
    
    # Generate training datasets
    print("\nGenerating training datasets...")
    generate_classification_data(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        output_file='train_classification.csv'
    )
    
    generate_regression_data(
        n_samples=1000,
        n_features=15,
        output_file='train_regression.csv'
    )
    
    # Generate test files for classification
    generate_test_files(n_files=3, base_name='test_classification', task='classification')
    
    # Generate test files for regression
    generate_test_files(n_files=3, base_name='test_regression', task='regression')
    
    print("\n" + "="*70)
    print("Data generation completed!")
    print("="*70)
    print("\nYou can now train models using:")
    print("  python main.py train --data train_classification.csv --target target --optimize --output model.pkl")
    print("\nAnd test using:")
    print("  python main.py test --model model.pkl --target target --test-files test_classification_*.csv")
