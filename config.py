"""
Configuration file for XGBoost model training and hyperparameter optimization.
"""

# Default hyperparameter search space for optimization
HYPERPARAMETER_GRID = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# Reduced grid for faster optimization (RandomizedSearchCV)
HYPERPARAMETER_GRID_SMALL = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5]
}

# Default XGBoost parameters
DEFAULT_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_jobs': -1
}

# Cross-validation settings
CV_FOLDS = 5

# Random search settings
RANDOM_SEARCH_ITERATIONS = 50
