#!/usr/bin/env python3
"""
XGBoost Model Training and Testing Application
Main CLI interface for training models with hyperparameter optimization
and testing on multiple files.
"""

import argparse
import sys
import os
from trainer import XGBoostTrainer
from tester import ModelTester


def train_model(args):
    """Train a model with optional hyperparameter optimization."""
    print("\n" + "="*70)
    print("XGBoost Model Training")
    print("="*70)
    
    # Initialize trainer
    trainer = XGBoostTrainer(task_type=args.task_type)
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data(
        filepath=args.data,
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Train model
    if args.optimize:
        # Hyperparameter optimization
        trainer.optimize_hyperparameters(
            X_train, y_train,
            method=args.optimize_method,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
            scoring=args.scoring
        )
    else:
        # Basic training
        trainer.train_basic(X_train, y_train)
    
    # Evaluate on test set
    metrics = trainer.evaluate(X_test, y_test)
    
    # Show feature importance
    if args.show_importance:
        trainer.get_feature_importance(top_n=args.top_features)
    
    # Save model
    if args.output:
        trainer.save_model(args.output, include_metadata=True)
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    
    return metrics


def test_model(args):
    """Test a trained model on multiple files."""
    print("\n" + "="*70)
    print("XGBoost Model Testing")
    print("="*70)
    
    # Initialize tester
    tester = ModelTester(model_path=args.model)
    
    # Get list of test files
    test_files = []
    for file_pattern in args.test_files:
        if os.path.isfile(file_pattern):
            test_files.append(file_pattern)
        elif os.path.isdir(file_pattern):
            # Add all supported files in directory
            for file in os.listdir(file_pattern):
                if file.endswith(('.csv', '.json', '.xlsx', '.xls', '.parquet')):
                    test_files.append(os.path.join(file_pattern, file))
        else:
            print(f"Warning: {file_pattern} is not a valid file or directory")
    
    if not test_files:
        print("Error: No valid test files found!")
        return
    
    # Test on multiple files
    results = tester.test_multiple_files(
        filepaths=test_files,
        target_column=args.target,
        save_report=True,
        report_path=args.report
    )
    
    print("\n" + "="*70)
    print("Testing completed successfully!")
    print("="*70)
    
    return results


def predict_single(args):
    """Make a prediction for a single sample."""
    print("\n" + "="*70)
    print("Single Sample Prediction")
    print("="*70)
    
    # Initialize tester
    tester = ModelTester(model_path=args.model)
    
    # Parse features from command line
    features = {}
    for feature_str in args.features:
        try:
            key, value = feature_str.split('=')
            # Try to convert to float, otherwise keep as string
            try:
                value = float(value)
            except ValueError:
                pass
            features[key] = value
        except ValueError:
            print(f"Warning: Invalid feature format '{feature_str}'. Use format: name=value")
            continue
    
    if not features:
        print("Error: No valid features provided!")
        return
    
    print(f"\nInput features: {features}")
    
    # Make prediction
    result = tester.predict_single_sample(features)
    
    print(f"\nPrediction result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    
    return result


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='XGBoost Model Training and Testing Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model with hyperparameter optimization
  python main.py train --data data.csv --target label --optimize --output model.pkl
  
  # Train a basic model without optimization
  python main.py train --data data.csv --target label --output model.pkl
  
  # Test a model on multiple files
  python main.py test --model model.pkl --target label --test-files test1.csv test2.csv
  
  # Test a model on all files in a directory
  python main.py test --model model.pkl --target label --test-files test_data/
  
  # Make a single prediction
  python main.py predict --model model.pkl --features feature1=5.1 feature2=3.5 feature3=1.4
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', required=True, help='Path to training data file')
    train_parser.add_argument('--target', required=True, help='Name of target column')
    train_parser.add_argument('--task-type', default='classification', 
                             choices=['classification', 'regression'],
                             help='Type of task (default: classification)')
    train_parser.add_argument('--optimize', action='store_true',
                             help='Enable hyperparameter optimization')
    train_parser.add_argument('--optimize-method', default='random',
                             choices=['random', 'grid'],
                             help='Optimization method (default: random)')
    train_parser.add_argument('--n-iter', type=int, default=50,
                             help='Number of iterations for random search (default: 50)')
    train_parser.add_argument('--cv-folds', type=int, default=5,
                             help='Number of cross-validation folds (default: 5)')
    train_parser.add_argument('--scoring', default='accuracy',
                             help='Scoring metric for optimization (default: accuracy)')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                             help='Proportion of data for testing (default: 0.2)')
    train_parser.add_argument('--random-state', type=int, default=42,
                             help='Random seed (default: 42)')
    train_parser.add_argument('--show-importance', action='store_true',
                             help='Show feature importance')
    train_parser.add_argument('--top-features', type=int, default=10,
                             help='Number of top features to show (default: 10)')
    train_parser.add_argument('--output', '-o', required=True,
                             help='Path to save the trained model')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a trained model')
    test_parser.add_argument('--model', required=True, help='Path to trained model file')
    test_parser.add_argument('--target', required=True, help='Name of target column')
    test_parser.add_argument('--test-files', nargs='+', required=True,
                            help='Paths to test files or directories')
    test_parser.add_argument('--report', default='test_report.json',
                            help='Path to save test report (default: test_report.json)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make a single prediction')
    predict_parser.add_argument('--model', required=True, help='Path to trained model file')
    predict_parser.add_argument('--features', nargs='+', required=True,
                               help='Features in format: name=value')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'train':
            train_model(args)
        elif args.command == 'test':
            test_model(args)
        elif args.command == 'predict':
            predict_single(args)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
