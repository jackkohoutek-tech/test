# Quick Start Guide

Get started with the XGBoost training app in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Generate Sample Data (Optional)

If you don't have your own data yet, generate sample datasets:

```bash
python example_data_generator.py
```

This creates:
- `train_classification.csv` - Training data for classification
- `train_regression.csv` - Training data for regression
- `test_classification_1.csv`, `test_classification_2.csv`, `test_classification_3.csv` - Test files
- `test_regression_1.csv`, `test_regression_2.csv`, `test_regression_3.csv` - Test files

## Step 3: Train Your First Model

### Option A: Quick Training (No Optimization)

```bash
python main.py train \
  --data train_classification.csv \
  --target target \
  --output my_model.pkl
```

### Option B: Train with Hyperparameter Optimization (Recommended)

```bash
python main.py train \
  --data train_classification.csv \
  --target target \
  --optimize \
  --show-importance \
  --output my_model.pkl
```

This will:
- Load and preprocess your data
- Perform hyperparameter optimization using RandomizedSearchCV
- Train the best model found
- Show feature importance
- Save the model to `my_model.pkl`

## Step 4: Test on Multiple Files

```bash
python main.py test \
  --model my_model.pkl \
  --target target \
  --test-files test_classification_*.csv
```

This will:
- Load your trained model
- Test on all specified files
- Print results for each file
- Generate a summary report
- Save detailed results to `test_report.json`

## Step 5: Make Predictions

```bash
python main.py predict \
  --model my_model.pkl \
  --features feature_0=1.5 feature_1=-0.8 feature_2=2.1
```

## Common Use Cases

### 1. Train a Regression Model

```bash
python main.py train \
  --data train_regression.csv \
  --target target \
  --task-type regression \
  --optimize \
  --scoring neg_mean_squared_error \
  --output regression_model.pkl
```

### 2. Fast Optimization (Fewer Iterations)

```bash
python main.py train \
  --data train_classification.csv \
  --target target \
  --optimize \
  --n-iter 20 \
  --output quick_model.pkl
```

### 3. Test All Files in a Directory

```bash
# Create a test directory
mkdir test_data
mv test_*.csv test_data/

# Test all files in the directory
python main.py test \
  --model my_model.pkl \
  --target target \
  --test-files test_data/
```

### 4. Custom Train/Test Split

```bash
python main.py train \
  --data train_classification.csv \
  --target target \
  --test-size 0.3 \
  --optimize \
  --output model_70_30.pkl
```

## Understanding the Output

### Training Output
- **Data shape**: Shows number of samples and features
- **Training/Test split**: Shows the size of each set
- **Optimization progress**: Real-time updates during hyperparameter search
- **Best parameters**: The optimal hyperparameters found
- **Evaluation metrics**: Performance on the test set
- **Feature importance**: Most predictive features

### Testing Output
- **Per-file results**: Metrics for each test file
- **Summary statistics**: Average performance across all files
- **Test report**: Detailed JSON report saved to disk

## Tips for Success

1. **Start Small**: Use `--n-iter 20` for faster initial experiments
2. **Check Feature Importance**: Use `--show-importance` to understand your model
3. **Monitor Performance**: Compare train vs. test metrics to detect overfitting
4. **Save Your Work**: Always use `--output` to save trained models
5. **Use Appropriate Metrics**: 
   - Classification: `accuracy`, `f1`, `roc_auc`
   - Regression: `neg_mean_squared_error`, `r2`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize hyperparameter grids in `config.py`
- Use the Python modules directly for more control (see Advanced Usage in README)
- Experiment with different `--cv-folds` values for better generalization

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### File Not Found
- Check that your file paths are correct
- Use absolute paths if relative paths don't work

### Target Column Not Found
- Verify the column name with: `head -1 your_data.csv`
- Column names are case-sensitive

### Poor Performance
- Try more optimization iterations: `--n-iter 100`
- Increase cross-validation folds: `--cv-folds 10`
- Check for data quality issues (missing values, outliers)
- Consider feature engineering

## Getting Help

```bash
# General help
python main.py --help

# Help for specific command
python main.py train --help
python main.py test --help
python main.py predict --help
```

Happy modeling! 🚀
