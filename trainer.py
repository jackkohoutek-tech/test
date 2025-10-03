"""
XGBoost Model Trainer with Hyperparameter Optimization.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config import (
    HYPERPARAMETER_GRID, HYPERPARAMETER_GRID_SMALL,
    DEFAULT_PARAMS, CV_FOLDS, RANDOM_SEARCH_ITERATIONS
)


class XGBoostTrainer:
    """
    A comprehensive XGBoost model trainer with hyperparameter optimization.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the trainer.
        
        Args:
            task_type: Type of task ('classification' or 'regression')
        """
        self.task_type = task_type
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.training_history = {}
        
    def load_data(self, filepath: str, target_column: str, 
                  test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Load data from file and split into train/test sets.
        
        Args:
            filepath: Path to the data file (CSV or other pandas-readable format)
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"Loading data from {filepath}...")
        
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
        print(f"Columns: {df.columns.tolist()}")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables
        X = self._preprocess_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
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
    
    def train_basic(self, X_train, y_train, params: Optional[Dict] = None) -> xgb.XGBClassifier:
        """
        Train a basic XGBoost model without hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            params: Optional custom parameters
            
        Returns:
            Trained XGBoost model
        """
        print("\n=== Training Basic Model ===")
        
        # Use default params if none provided
        if params is None:
            params = DEFAULT_PARAMS.copy()
        
        # Adjust objective based on task type
        if self.task_type == 'classification':
            # Check if binary or multiclass
            n_classes = len(np.unique(y_train))
            if n_classes == 2:
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
            else:
                params['objective'] = 'multi:softprob'
                params['num_class'] = n_classes
                params['eval_metric'] = 'mlogloss'
            
            self.model = xgb.XGBClassifier(**params)
        else:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
            self.model = xgb.XGBRegressor(**params)
        
        # Train model
        self.model.fit(X_train, y_train, verbose=False)
        
        self.best_params = params
        print("Training completed!")
        
        return self.model
    
    def optimize_hyperparameters(self, X_train, y_train, 
                                 method: str = 'random',
                                 n_iter: int = RANDOM_SEARCH_ITERATIONS,
                                 cv_folds: int = CV_FOLDS,
                                 scoring: str = 'accuracy') -> Dict:
        """
        Optimize hyperparameters using Grid Search or Random Search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: 'random' or 'grid'
            n_iter: Number of iterations for random search
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary of best parameters
        """
        print(f"\n=== Hyperparameter Optimization ({method.upper()}) ===")
        print(f"Cross-validation folds: {cv_folds}")
        
        # Initialize base model
        if self.task_type == 'classification':
            n_classes = len(np.unique(y_train))
            if n_classes == 2:
                base_model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=42,
                    n_jobs=-1
                )
            else:
                base_model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=n_classes,
                    random_state=42,
                    n_jobs=-1
                )
        else:
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
        
        # Choose search method
        if method == 'random':
            print(f"Running RandomizedSearchCV with {n_iter} iterations...")
            search = RandomizedSearchCV(
                base_model,
                param_distributions=HYPERPARAMETER_GRID_SMALL,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        elif method == 'grid':
            print("Running GridSearchCV (this may take a while)...")
            search = GridSearchCV(
                base_model,
                param_grid=HYPERPARAMETER_GRID_SMALL,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'random' or 'grid'")
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Store results
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        
        print(f"\nBest score: {search.best_score_:.4f}")
        print("Best parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Store training history
        self.training_history = {
            'best_score': search.best_score_,
            'best_params': self.best_params,
            'cv_results': search.cv_results_
        }
        
        return self.best_params
    
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n=== Model Evaluation ===")
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_basic() or optimize_hyperparameters() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        metrics = {}
        
        if self.task_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            
            print(f"MSE:  {metrics['mse']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE:  {metrics['mae']:.4f}")
            print(f"R²:   {metrics['r2']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else range(len(importance)),
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Feature Importances:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    
    def save_model(self, filepath: str, include_metadata: bool = True):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            include_metadata: Whether to save metadata alongside the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        print(f"\nSaving model to {filepath}...")
        
        # Save model
        joblib.dump(self.model, filepath)
        
        # Save metadata
        if include_metadata:
            metadata = {
                'task_type': self.task_type,
                'best_params': self.best_params,
                'feature_names': self.feature_names,
                'training_date': datetime.now().isoformat()
            }
            
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved to {metadata_path}")
        
        print("Model saved successfully!")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        print(f"Loading model from {filepath}...")
        
        self.model = joblib.load(filepath)
        
        # Try to load metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.task_type = metadata.get('task_type', 'classification')
                self.best_params = metadata.get('best_params')
                self.feature_names = metadata.get('feature_names')
            print(f"Metadata loaded from {metadata_path}")
        
        print("Model loaded successfully!")
