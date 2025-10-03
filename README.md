# XGBoost Model Training and Testing Application

A comprehensive application for training XGBoost models with hyperparameter optimization and testing on multiple files.

## Features

- **XGBoost Model Training**: Train classification and regression models
- **Hyperparameter Optimization**: Automated hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- **Multi-file Testing**: Test trained models on multiple files and generate comprehensive reports
- **Feature Importance Analysis**: Visualize which features matter most
- **Model Persistence**: Save and load trained models with metadata
- **CLI Interface**: Easy-to-use command-line interface
- **Support for Multiple File Formats**: CSV, Excel, JSON, Parquet

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The application provides three main commands: `train`, `test`, and `predict`.

### 1. Training a Model

#### Basic Training (No Optimization)

```bash
python main.py train \
  --data data.csv \
  --target label \
  --output model.pkl
```

#### Training with Hyperparameter Optimization

```bash
python main.py train \
  --data data.csv \
  --target label \
  --optimize \
  --optimize-method random \
  --n-iter 50 \
  --cv-folds 5 \
  --output model.pkl
```

#### Training Options

- `--data`: Path to training data file (required)
- `--target`: Name of the target column (required)
- `--output`: Path to save the trained model (required)
- `--task-type`: Task type - `classification` or `regression` (default: classification)
- `--optimize`: Enable hyperparameter optimization
- `--optimize-method`: Optimization method - `random` or `grid` (default: random)
- `--n-iter`: Number of iterations for random search (default: 50)
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--scoring`: Scoring metric for optimization (default: accuracy)
- `--test-size`: Proportion of data for testing (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--show-importance`: Display feature importance
- `--top-features`: Number of top features to display (default: 10)

### 2. Testing a Model

#### Test on Multiple Files

```bash
python main.py test \
  --model model.pkl \
  --target label \
  --test-files test1.csv test2.csv test3.csv
```

#### Test on All Files in a Directory

```bash
python main.py test \
  --model model.pkl \
  --target label \
  --test-files test_data/
```

#### Testing Options

- `--model`: Path to trained model file (required)
- `--target`: Name of the target column (required)
- `--test-files`: Paths to test files or directories (required, can specify multiple)
- `--report`: Path to save the test report JSON (default: test_report.json)

### 3. Making Single Predictions

```bash
python main.py predict \
  --model model.pkl \
  --features feature1=5.1 feature2=3.5 feature3=1.4 feature4=0.2
```

#### Prediction Options

- `--model`: Path to trained model file (required)
- `--features`: Features in format `name=value` (required, can specify multiple)

## Project Structure

```
.
├── main.py              # Main CLI interface
├── trainer.py           # Model training and optimization module
├── tester.py            # Model testing module
├── config.py            # Configuration and hyperparameter grids
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration

You can customize hyperparameter search spaces in `config.py`:

- `HYPERPARAMETER_GRID`: Full grid for exhaustive search
- `HYPERPARAMETER_GRID_SMALL`: Reduced grid for faster optimization
- `DEFAULT_PARAMS`: Default XGBoost parameters
- `CV_FOLDS`: Number of cross-validation folds
- `RANDOM_SEARCH_ITERATIONS`: Default number of random search iterations

## Examples

### Example 1: Train a Classification Model

```bash
# Train with hyperparameter optimization
python main.py train \
  --data iris.csv \
  --target species \
  --task-type classification \
  --optimize \
  --optimize-method random \
  --n-iter 30 \
  --show-importance \
  --output iris_model.pkl
```

### Example 2: Train a Regression Model

```bash
# Train a regression model
python main.py train \
  --data housing.csv \
  --target price \
  --task-type regression \
  --optimize \
  --scoring neg_mean_squared_error \
  --output housing_model.pkl
```

### Example 3: Test on Multiple Files

```bash
# Test the trained model on multiple test sets
python main.py test \
  --model iris_model.pkl \
  --target species \
  --test-files test_set1.csv test_set2.csv test_set3.csv \
  --report iris_test_results.json
```

### Example 4: Batch Testing

```bash
# Test on all CSV files in a directory
python main.py test \
  --model model.pkl \
  --target label \
  --test-files test_data/ \
  --report batch_results.json
```

## Supported File Formats

- **CSV**: `.csv`
- **Excel**: `.xlsx`, `.xls`
- **JSON**: `.json`
- **Parquet**: `.parquet`

## Model Output

When training a model, the application saves:

1. **Model file** (`.pkl`): The trained XGBoost model
2. **Metadata file** (`_metadata.json`): Contains:
   - Task type (classification/regression)
   - Best hyperparameters
   - Feature names
   - Training date

## Test Report

The test report (JSON format) includes:

- Model path and test date
- Task type
- Results for each test file:
  - Performance metrics (accuracy, precision, recall, F1, etc.)
  - Confusion matrix (for classification)
  - Error metrics (for regression)
  - Number of samples
- Summary statistics across all files

## Performance Metrics

### Classification Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- ROC AUC (binary classification only)
- Confusion Matrix
- Classification Report

### Regression Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Advanced Usage

### Using as a Python Module

```python
from trainer import XGBoostTrainer
from tester import ModelTester

# Train a model
trainer = XGBoostTrainer(task_type='classification')
X_train, X_test, y_train, y_test = trainer.load_data('data.csv', 'target')
trainer.optimize_hyperparameters(X_train, y_train, method='random')
trainer.evaluate(X_test, y_test)
trainer.save_model('model.pkl')

# Test the model
tester = ModelTester('model.pkl')
results = tester.test_multiple_files(['test1.csv', 'test2.csv'], 'target')
```

## Tips for Better Results

1. **Start with Random Search**: It's faster and often finds good solutions
2. **Use Cross-Validation**: More folds = better estimates but slower training
3. **Feature Engineering**: Better features = better models
4. **Handle Imbalanced Data**: Consider using appropriate metrics and sampling techniques
5. **Monitor Overfitting**: Check train vs. test performance

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **File not found**: Check file paths and ensure files exist
3. **Target column not found**: Verify the target column name matches your data
4. **Out of memory**: Reduce dataset size or use fewer hyperparameter combinations

## License

This project is open source and available for educational and commercial use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Future Enhancements

- [ ] Add support for custom evaluation metrics
- [ ] Implement feature selection
- [ ] Add visualization of results
- [ ] Support for distributed training
- [ ] Integration with MLflow for experiment tracking
- [ ] Add support for early stopping
- [ ] Implement ensemble methods
