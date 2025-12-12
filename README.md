# Predictive Modeling for Powerlifting Performance Using Regression ML Techniques

## Overview

This project implements predictive modeling techniques to forecast powerlifting performance using the **Wilks Score** as the target variable. The Wilks Score is a standardized measure that normalizes a lifter's strength relative to their bodyweight, enabling fair comparisons across different weight classes and genders.

## Dataset

- **Total Samples**: 1,981 powerlifters
- **Features**: 9 attributes (8 continuous, 1 categorical)
- **Target Variable**: Wilks_Score

### Features

**Continuous Variables:**
- `Age`: Lifter's age
- `Powerlifting_Exp_Years`: Years of powerlifting experience
- `Bodyweight_kg`: Body weight in kilograms
- `Squat_PR_kg`: Personal record in squat (kg)
- `Deadlift_PR_kg`: Personal record in deadlift (kg)
- `Bench_PR_kg`: Personal record in bench press (kg)
- `Height_cm`: Height in centimeters

**Categorical Variables:**
- `Gender`: Encoded as strength coefficients (Female: 0.8, Male: 1.0) to reflect biological differences relevant to Wilks Score calculation

## Key Findings

### Feature Correlations
- **Highest correlation with Wilks_Score**: `Powerlifting_Exp_Years` (0.62) and `Squat_PR_kg` (0.47)
- **Inter-feature correlations**: Strong correlations between `Squat_PR_kg`, `Bench_PR_kg`, and `Deadlift_PR_kg`

### Data Quality
- No missing values
- Features follow approximately normal distributions
- No significant skewness or irregularities detected

## Methodology

### 1. Data Preprocessing
- Gender encoding using strength coefficients (0.8 for Female, 1.0 for Male)
- StandardScaler normalization for all features
- Train-test split (80-20)

### 2. Exploratory Data Analysis
- Statistical summary and distribution analysis
- Correlation heatmap visualization
- Feature-target relationship analysis
- Comprehensive data profiling using `ydata_profiling`

### 3. Regression Models

#### Linear Regression
- Baseline model with closed-form solution (normal equation)
- Cross-validation with 4-fold KFold
- **Results**: CV RMSE: 14.96 ± 0.30

#### SGD Regressor (Stochastic Gradient Descent)
- Hyperparameter tuning across multiple dimensions:
  - **Regularization**: L1, L2, Elastic Net
  - **Learning rate schedules**: `invscaling`, `optimal`
  - **Initial learning rates**: 0.001, 0.01, 0.05
  - **Regularization strength (α)**: 1e-5 to 1e-1

**Best Configuration:**
- Penalty: Elastic Net (α=0.0001, l1_ratio=0.3)
- Learning rate: `invscaling` with η₀=0.001
- **Best CV RMSE**: ~14.97

### 4. Model Evaluation

**Metrics:**
- Root Mean Squared Error (RMSE)
- Cross-validation RMSE with standard deviation
- Learning curves (training vs validation)

**Key Insights:**
- **L2 Regularization**: Stable but slightly higher RMSE (mild underfitting)
- **L1 Regularization**: Higher variance, some useful predictors removed
- **Elastic Net**: Best performance, balancing sparsity and smooth regularization
- Optimal learning rate schedule (`invscaling`) ensures smooth convergence

## Technologies Used

- **Python 3.x**
- **NumPy & Pandas**: Data manipulation
- **Scikit-learn**: Machine learning models and utilities
  - `LinearRegression`
  - `SGDRegressor`
  - `StandardScaler`
  - `KFold`, `cross_val_score`
- **Matplotlib & Seaborn**: Data visualization
- **ydata_profiling**: Comprehensive data profiling

## Installation

```bash
# Clone the repository
git clone https://github.com/prabha-07/Predictive-Modeling-for-Powerlifting-Performance-Using-Regression-ML-techniques.git

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn ydata-profiling
```

## Usage

1. **Prepare your dataset**: Ensure your CSV file follows the same structure as described above
2. **Update the file path** in the notebook: Modify the path in `pd.read_csv()` to point to your dataset
3. **Run the notebook**: Execute cells sequentially to:
   - Load and explore the data
   - Preprocess features
   - Train regression models
   - Evaluate performance

## Results Summary

| Model | CV RMSE | Train RMSE | Test RMSE |
|-------|---------|------------|-----------|
| Linear Regression | 14.96 ± 0.30 | 14.88 | 16.06 |
| SGD Regressor (Best) | ~14.97 | - | - |

## Key Contributions

- Custom powerlifting dataset creation with Wilks Score as target
- Comprehensive hyperparameter tuning for SGD Regressor
- Detailed analysis of regularization impact (L1, L2, Elastic Net)
- Learning curve visualization for model convergence analysis
- Gender encoding using strength coefficients relevant to powerlifting

## Future Enhancements

- Polynomial feature engineering
- Additional regression models (Ridge, Lasso, Elastic Net)
- Feature importance analysis
- Residual analysis and diagnostics
- Model interpretation and explainability

## License

This project is open source and available for educational purposes.

## Author

**prabha-07**

---

*For questions or contributions, please open an issue or submit a pull request.*

