"""
Model Testing Module - Test trained models on multiple files.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


class ModelTester:
    """
    Test trained models on multiple test files and generate comprehensive reports.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the tester with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.task_type = 'classification'
        self.feature_names = None
        self.metadata = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and metadata from disk."""
        print(f"Loading model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model
        self.model = joblib.load(self.model_path)
        
        # Load metadata if available
        metadata_path = self.model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.task_type = self.metadata.get('task_type', 'classification')
                self.feature_names = self.metadata.get('feature_names')
            print(f"Metadata loaded: Task type = {self.task_type}")
        else:
            print("Warning: No metadata file found. Assuming classification task.")
        
        print("Model loaded successfully!")
    
    def _load_data(self, filepath: str, target_column: str) -> tuple:
        """
        Load data from a file.
        
        Args:
            filepath: Path to the data file
            target_column: Name of the target column
            
        Returns:
            Tuple of (X, y)
        """
        print(f"\nLoading data from {filepath}...")
        
        # Load data based on file extension
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        print(f"Data shape: {df.shape}")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Preprocess features
        X = self._preprocess_features(X)
        
        return X, y
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features (handle categorical variables, missing values, etc.)
        
        Args:
            X: Feature dataframe
            
        Returns:
            Preprocessed feature dataframe
        """
        X = X.copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                X[col] = X[col].fillna('missing')
            else:
                X[col] = X[col].fillna(X[col].median())
        
        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                X[col] = pd.Categorical(X[col]).codes
        
        return X
    
    def test_single_file(self, filepath: str, target_column: str) -> Dict[str, Any]:
        """
        Test the model on a single file.
        
        Args:
            filepath: Path to the test file
            target_column: Name of the target column
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load data
        X, y = self._load_data(filepath, target_column)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        metrics = {
            'filename': os.path.basename(filepath),
            'num_samples': len(y)
        }
        
        if self.task_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC for binary classification
            if len(np.unique(y)) == 2:
                try:
                    y_pred_proba = self.model.predict_proba(X)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
                except:
                    metrics['roc_auc'] = None
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
            
            # Classification report
            metrics['classification_report'] = classification_report(y, y_pred, output_dict=True)
            
            print(f"\n{'='*60}")
            print(f"Results for: {os.path.basename(filepath)}")
            print(f"{'='*60}")
            print(f"Samples:   {metrics['num_samples']}")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1_score']:.4f}")
            if metrics.get('roc_auc'):
                print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
            
        else:
            # Regression metrics
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y, y_pred)
            metrics['r2'] = r2_score(y, y_pred)
            
            print(f"\n{'='*60}")
            print(f"Results for: {os.path.basename(filepath)}")
            print(f"{'='*60}")
            print(f"Samples: {metrics['num_samples']}")
            print(f"MSE:     {metrics['mse']:.4f}")
            print(f"RMSE:    {metrics['rmse']:.4f}")
            print(f"MAE:     {metrics['mae']:.4f}")
            print(f"R²:      {metrics['r2']:.4f}")
        
        return metrics
    
    def test_multiple_files(self, filepaths: List[str], target_column: str,
                           save_report: bool = True,
                           report_path: str = 'test_report.json') -> List[Dict[str, Any]]:
        """
        Test the model on multiple files and generate a comprehensive report.
        
        Args:
            filepaths: List of paths to test files
            target_column: Name of the target column
            save_report: Whether to save the report to disk
            report_path: Path to save the report
            
        Returns:
            List of metrics dictionaries for each file
        """
        print(f"\n{'#'*60}")
        print(f"Testing model on {len(filepaths)} files")
        print(f"{'#'*60}")
        
        all_metrics = []
        
        for filepath in filepaths:
            try:
                metrics = self.test_single_file(filepath, target_column)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"\nError testing {filepath}: {str(e)}")
                all_metrics.append({
                    'filename': os.path.basename(filepath),
                    'error': str(e)
                })
        
        # Generate summary
        self._print_summary(all_metrics)
        
        # Save report
        if save_report:
            self._save_report(all_metrics, report_path)
        
        return all_metrics
    
    def _print_summary(self, all_metrics: List[Dict[str, Any]]):
        """Print a summary of results across all files."""
        print(f"\n{'#'*60}")
        print("SUMMARY")
        print(f"{'#'*60}")
        
        # Filter out error results
        valid_metrics = [m for m in all_metrics if 'error' not in m]
        error_count = len(all_metrics) - len(valid_metrics)
        
        print(f"Total files tested: {len(all_metrics)}")
        print(f"Successful tests:   {len(valid_metrics)}")
        print(f"Failed tests:       {error_count}")
        
        if valid_metrics:
            if self.task_type == 'classification':
                avg_accuracy = np.mean([m['accuracy'] for m in valid_metrics])
                avg_precision = np.mean([m['precision'] for m in valid_metrics])
                avg_recall = np.mean([m['recall'] for m in valid_metrics])
                avg_f1 = np.mean([m['f1_score'] for m in valid_metrics])
                
                print(f"\nAverage Metrics:")
                print(f"  Accuracy:  {avg_accuracy:.4f}")
                print(f"  Precision: {avg_precision:.4f}")
                print(f"  Recall:    {avg_recall:.4f}")
                print(f"  F1 Score:  {avg_f1:.4f}")
                
                # Check for ROC AUC
                roc_aucs = [m.get('roc_auc') for m in valid_metrics if m.get('roc_auc')]
                if roc_aucs:
                    print(f"  ROC AUC:   {np.mean(roc_aucs):.4f}")
            else:
                avg_rmse = np.mean([m['rmse'] for m in valid_metrics])
                avg_mae = np.mean([m['mae'] for m in valid_metrics])
                avg_r2 = np.mean([m['r2'] for m in valid_metrics])
                
                print(f"\nAverage Metrics:")
                print(f"  RMSE: {avg_rmse:.4f}")
                print(f"  MAE:  {avg_mae:.4f}")
                print(f"  R²:   {avg_r2:.4f}")
    
    def _save_report(self, all_metrics: List[Dict[str, Any]], report_path: str):
        """Save the test report to disk."""
        report = {
            'model_path': self.model_path,
            'test_date': datetime.now().isoformat(),
            'task_type': self.task_type,
            'results': all_metrics
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTest report saved to: {report_path}")
    
    def predict_single_sample(self, features: Dict[str, Any]) -> Any:
        """
        Make a prediction for a single sample.
        
        Args:
            features: Dictionary of feature names and values
            
        Returns:
            Prediction result
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Preprocess
        df = self._preprocess_features(df)
        
        # Predict
        prediction = self.model.predict(df)[0]
        
        # Get probability for classification
        if self.task_type == 'classification':
            try:
                proba = self.model.predict_proba(df)[0]
                return {
                    'prediction': int(prediction),
                    'probabilities': proba.tolist()
                }
            except:
                return {'prediction': int(prediction)}
        else:
            return {'prediction': float(prediction)}
