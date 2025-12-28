
# Car Price Prediction with Linear Regression (from Scratch)

This project implements **linear regression from scratch** to predict used car prices based on technical and categorical features . It includes a closed-form solution, stochastic gradient descent (SGD) variants, polynomial regression, and comparisons with scikit-learn and statsmodels implementations .

## Features

- Custom implementation of:
  - Linear regression (closed-form normal equation)
  - Polynomial regression (closed-form)
  - Mini-batch SGD
  - Adaptive optimizers: AdaGrad, RMSProp, Adam 
- Data pipeline for real-world tabular data:
  - Cleaning messy numeric strings (e.g. engine size, power fields)
  - Handling missing values
  - Encoding categorical variables (frequency + one-hot)
  - Feature scaling and zero-variance feature removal 
- Exploratory Data Analysis (EDA):
  - Descriptive statistics
  - Histograms and boxplots
  - Correlation heatmap 
- Model comparison with:
  - scikit-learn `LinearRegression`
  - statsmodels `OLS` 

> Note: Some experimental parts (especially **SGD on polynomial features** and some optimizer settings) are intentionally left in a “work-in-progress” state and can be improved (see **Limitations & TODO** below) .

## Dataset

The dataset is a used car listing dataset stored in `data/data.csv` .  
Example columns:

- Categorical:
  - `Make`, `Model`, `Fuel Type`, `Transmission`, `Location`, `Color`, `Owner`, `Seller Type`, `Engine`, `Drivetrain` 
- Numerical:
  - `Price`, `Year`, `Kilometer`, `Length`, `Width`, `Height`, `Seating Capacity`, `Fuel Tank Capacity`, `max_power_bhp`, `max_power_rpm`, `max_torque_Nm`, `max_torque_rpm` 

Target:

- `Price` (continuous), predicted from the other features .

## Project Structure

```
ml-from-scratch/
└── linear_regression/
    ├── data/
    │   └── data.csv
    ├── main.py
    └── src/
        ├── dataloader.py
        ├── preprocessor.py
        ├── metrics.py
        ├── linearregression.py
        ├── sgdregressor.py
        └── __init__.py 
```


- `dataloader.py`  
  Loads the CSV, cleans numeric-like columns (e.g. extracting numbers from strings) .
- `preprocessor.py`  
  Handles missing values, train–test split, scaling, bias term, polynomial feature generation, and categorical encoding .
- `metrics.py`  
  Implements MSE, RMSE, MAE, and R² .
- `linearregression.py`  
  Closed-form linear and polynomial regression .
- `sgdregressor.py`  
  SGD, AdaGrad, RMSProp, Adam implementations for linear regression .
- `main.py`  
  Orchestrates the full pipeline: EDA → preprocessing → models → optimizers → comparison with libraries .

## Methods

### 1. Closed-Form Linear Regression

- Adds a bias column.
- Solves \( w = (X^T X)^{+} X^T y \) via pseudo-inverse for numerical stability .
- On this dataset, the closed-form model achieves roughly:

  - `MSE ≈ 7.61e11`
  - `RMSE ≈ 872,124`
  - `MAE ≈ 614,736`
  - `R² ≈ 0.798`  
    (very close to scikit-learn and statsmodels results) .

### 2. SGD and Mini-batch Experiments

- Evaluates `AdaptiveSGD` with different batch sizes: `[4, 8, 16, 32, 64, 128]` .
- For this configuration, SGD **diverges** (huge MSE and negative R²) due to:
  - Large learning rate for this scale of target values
  - No target scaling
  - Limited number of epochs .

Example output snapshot:

```
=== Running SGD with batch size 4 ===
Time: 0.24s | MSE=5.84e12 | RMSE=2.42e6 | MAE=1.44e6 | R2=-0.55
...
```


### 3. Polynomial Regression

- Polynomial feature generator:
  - Original features
  - Squared terms
  - Pairwise interaction terms .
- Closed-form polynomial regression achieves significantly better fit:

  - `MSE ≈ 3.71e11`
  - `RMSE ≈ 609,186`
  - `MAE ≈ 419,839`
  - `R² ≈ 0.90` .

- **SGD with polynomial features** currently **blows up**:
  - `MSE = inf`, `RMSE = inf`, `R² = -inf`, extremely large MAE .
  - This is a known limitation (learning rate & feature scaling not tuned for high-degree polynomial space) .

### 4. Optimizers: AdaGrad, RMSProp, Adam

- Implemented as subclasses of `SGDRegressor` with different update rules .
- Currently used on the linear feature space (not polynomial) and track training loss curves per epoch .
- Runtime warnings in metrics (overflow in MSE) indicate that for some runs, predictions explode and need better hyperparameter tuning and/or target normalization .

### 5. Library Baselines

- **scikit-learn `LinearRegression`**
  - Very similar performance to custom closed-form implementation:

  ```
  MSE:  760601832891.796
  RMSE: 872124.895
  MAE:  614736.377
  R2:   0.7978
  ```
  

- **statsmodels OLS**
  - Same metrics as above, with regression summary available if needed .

## How to Run

1. Make sure the dataset is under `linear_regression/data/data.csv`:

   ```
   ml-from-scratch/
   └── linear_regression/
       ├── data/
       │   └── data.csv
       ├── main.py
       └── src/...
   ```
   

2. Install dependencies:

   ```
   pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
   ```
   

3. From the `linear_regression` directory, run:

   ```
   python main.py
   ```
   

This will:

- Print data types before/after preprocessing
- Run EDA (histograms, boxplots, correlation heatmap)
- Train and evaluate:
  - Closed-form linear regression
  - SGD experiments
  - Polynomial regression (closed-form + SGD)
  - AdaGrad, RMSProp, Adam
  - scikit-learn and statsmodels baselines .

## Limitations & TODO

Some parts are intentionally left as **work-in-progress** to experiment with stability and optimization:

- **SGD on original features**
  - Current learning rate and lack of target scaling lead to divergence and very high MSE/R² .
  - TODO:
    - Lower learning rate
    - Normalize/standardize `Price`
    - Increase epochs and add early stopping .

- **SGD on polynomial features**
  - Currently completely unstable (MSE and R² become infinite) .
  - TODO:
    - Stronger regularization and smaller learning rate
    - Better feature scaling (possibly separate scaling for polynomial features)
    - Gradient clipping .

- **Runtime warnings (overflow in metrics)**
  - Caused by extremely large predictions when training diverges .
  - TODO:
    - Add checks for NaN / inf in loss
    - Implement safe guards in training loop .

These open issues make the project a good playground for experimenting with **numerical stability, optimization, and regularization in regression** .

## Author

Developed by **Fateme Rezvan (2025)** as a learning project on linear regression, stochastic optimization, and practical issues in training regression models on real-world tabular data .
```

***

