
# Logistic Regression (Binary & Multiclass) from Scratch

This project implements **logistic regression from scratch** in NumPy for both binary and multiclass classification and compares it with a custom KNN classifier . It is evaluated on the Breast Cancer Wisconsin dataset (binary) and the Wine Quality (white) dataset (multiclass) and includes full preprocessing, training loops, loss curves, and metric calculations .

---

## Project Goals

- Implement logistic regression **without** using high-level ML libraries (no scikit-learn `LogisticRegression`) .
- Support:
  - Binary logistic regression with sigmoid and binary cross-entropy.
  - Multiclass (softmax) logistic regression with categorical cross-entropy .
- Compare performance with:
  - A custom KNN implementation (binary task).
- Provide a clean **training pipeline**:
  - Data loading and preprocessing
  - Train/test split and feature scaling
  - Loss tracking and visualization
  - Evaluation with common classification metrics .

---

## Datasets

### 1. Breast Cancer (Binary Classification)

- File: `data/wdbc.csv` .
- Target column: `diagnosis`
  - `M` → malignant → mapped to `1`
  - `B` → benign → mapped to `0` .
- Features: numeric descriptors of cell nuclei (e.g. radius, texture, perimeter, area, etc.) .
- Task: classify each sample as malignant or benign .

### 2. Wine Quality – White (Multiclass Classification)

- File: `data/winequality-white.csv` with `;` separator .
- Target column: `quality` (originally 3–9) .
- In the code, labels are shifted: `y = quality - 3` resulting in **classes 0..5** .
- Features: physicochemical properties such as acidity, residual sugar, chlorides, density, alcohol, etc. .
- Task: predict quality class of each wine sample .

---

## Code Structure

```
ml-from-scratch/
└── logistic_regression/
    ├── data/
    │   ├── wdbc.csv
    │   └── winequality-white.csv
    ├── main.py
    └── src/
        ├── preprocessor.py
        ├── logistic_regression.py
        ├── knn.py
        └── evaluator.py
```


### `src/preprocessor.py`

- `load_data(filepath, sep=None)`  
  Loads CSV files with optional custom separator .
- `clean_data()` (for cancer dataset)  
  Maps `diagnosis` from `{M, B}` to `{1, 0}` .
- `split(X, y, test_size=0.2, random_state=None)`  
  Manual random shuffle and train/test split using NumPy indexing .
- `scale(X_train, X_test)`  
  Standardizes features using training mean and standard deviation, with protection against extremely small std .

(Encoder class is defined but not used in this project; it is available for future categorical encoding extensions .)

### `src/logistic_regression.py`

Implements logistic regression with gradient descent:

- Parameters:
  - `lr`: learning rate
  - `epochs`: number of full-batch iterations
  - `multi_class`: `False` for binary, `True` for softmax .
- Utilities:
  - `sigmoid(z)` for binary logistic regression .
  - `softmax(z)` with max-subtraction for numerical stability .
  - `one_hot(y, C)` for multiclass labels .
- Loss functions:
  - Binary cross-entropy:  
    \[
    L = -\frac{1}{n} \sum_i y_i \log(\hat y_i) + (1 - y_i)\log(1 - \hat y_i)
    \] .
  - Categorical cross-entropy for one-hot labels and softmax probabilities .
- Training:
  - Adds a bias column of ones to `X` .
  - Binary:
    - Weight vector `W` with shape `(d+1,)`.
    - Forward pass: `z = X @ W`, `y_hat = sigmoid(z)`.
    - Gradient: `grad = X.T @ (y_hat - y) / n`.
  - Multiclass:
    - Weight matrix `W` with shape `(d+1, C)`.
    - Forward pass: `scores = X @ W`, `probs = softmax(scores)`.
    - Gradient: `grad = X.T @ (probs - y_one_hot) / n` .
  - After each epoch, the loss is appended to `loss_history` for plotting .
- Prediction:
  - Binary: threshold sigmoid probabilities at 0.5 .
  - Multiclass: `argmax` over class scores .

### `src/knn.py`

- Simple KNN classifier with Euclidean distance and brute-force neighbor search .
- Used only on the binary breast cancer task to compare with logistic regression .

### `src/evaluator.py`

- Builds confusion matrices for arbitrary number of classes .
- Metrics:
  - Accuracy
  - Precision/Recall/F1 per class
  - Macro-F1 (average of class-wise F1) .
- `evaluate_binary` returns accuracy, precision, recall, and F1 for the positive class (class `1`) .

---

## Experiments and Results

### 1. Binary Logistic Regression vs KNN (Breast Cancer)

Configuration:

- Learning rate: `lr = 0.01`
- Epochs: `3000`
- Features: normalized numeric columns from `wdbc.csv` .

Results:

```
--- Logistic Regression (Binary) ---
accuracy: 0.9649
precision: 0.9778
recall: 0.9362
f1: 0.9565
Final Training Loss: 0.0674
```


- Loss curve: binary cross-entropy decreases smoothly from around 0.7 to ~0.067 over 3000 epochs.  
- The model achieves high precision and recall, with F1 ≈ 0.96, indicating a strong balance between detecting malignant cases and avoiding false positives .

KNN comparison (same train/test split and scaled features):

```
k = 1  → Accuracy ≈ 0.947, F1 ≈ 0.935
k = 3  → Accuracy ≈ 0.947, F1 ≈ 0.933
k = 5  → Accuracy ≈ 0.930, F1 ≈ 0.909
k = 7  → Accuracy ≈ 0.930, F1 ≈ 0.909
k = 9  → Accuracy ≈ 0.930, F1 ≈ 0.909
k = 11 → Accuracy ≈ 0.939, F1 ≈ 0.920
```


- Logistic regression slightly outperforms KNN for this task, both in accuracy and F1-score .
- A plot of `k` vs `F1-score` is generated to visualize KNN performance over different neighborhood sizes .

### 2. Multiclass Softmax Logistic Regression (Wine Quality)

Configuration:

- Learning rate: `lr = 0.05`
- Epochs: `4000`
- Labels: `quality - 3`, giving 6 classes (0..5) .
- Softmax with full-batch gradient descent .

Results:

```
--- Logistic Regression (Multiclass) ---
Accuracy : 0.5265
Macro F1 : 0.2236
Final training loss: 1.0919
```


- The training loss curve shows a clear decreasing trend and stabilizes around 1.09, indicating that optimization converges .
- However, macro F1 is relatively low (~0.22), suggesting that some minority classes are not predicted well and there is **class imbalance and/or limited model capacity** in this simple linear classifier .

---

## Visualizations

The following plots are generated:

- **Binary logistic regression loss curve**  
  Binary cross-entropy vs epoch (smooth exponential-like decay).
- **Softmax logistic regression loss curve**  
  Categorical cross-entropy vs epoch for the wine quality dataset.
- **k vs F1-score for KNN (Breast Cancer)**  
  Line plot showing how F1 varies with k .

These figures clearly show convergence of the gradient descent training process and highlight trade-offs between models and hyperparameters .

---

## How to Run

1. Place the datasets in the `data` folder:

   ```
   logistic_regression/
   ├── data/
   │   ├── wdbc.csv
   │   └── winequality-white.csv
   ├── main.py
   └── src/...
   ```
   

2. Install dependencies:

   ```
   pip install numpy pandas matplotlib
   ```
   (Seaborn / scikit-learn are not required for the core logistic regression, only NumPy and Matplotlib are used here.) 

3. From the `logistic_regression` directory, run:

   ```
   python main.py
   ```
   

4. The script will:
   - Train and evaluate binary logistic regression and KNN on the breast cancer dataset.
   - Train and evaluate multiclass softmax logistic regression on the wine quality dataset.
   - Print metrics to the console and show the loss curves as Matplotlib figures .

---

## Limitations and Future Work

- **Multiclass performance**: Accuracy ≈ 0.53 and macro F1 ≈ 0.22 indicate that some classes are poorly predicted, likely due to:
  - Class imbalance in wine quality labels.
  - Linear decision boundaries of logistic regression.
  - Lack of regularization and hyperparameter search .

- **Hyperparameter tuning**:
  - Current learning rates (`0.01` for binary, `0.05` for multiclass) and epoch counts (3000–4000) were chosen heuristically .
  - Future work:
    - Implement validation set and grid search for `lr` and `epochs`.
    - Add L2 regularization to improve generalization .

- **KNN optimization**:
  - KNN uses a simple brute-force search and could be improved with KD-trees or ball trees for larger datasets .

- **Preprocessing generalization**:
  - Currently preprocessing for cancer vs wine datasets is slightly different; a more generic pipeline could unify them and support more datasets with minimal changes .

---

## Academic Perspective

From a university / educational viewpoint, this project demonstrates:

- The derivation and implementation of **logistic regression** as a probabilistic linear classifier for both binary and multiclass settings .
- Practical aspects of **gradient-based optimization**, including:
  - Choice of learning rate.
  - Monitoring convergence via loss curves.
  - Numerical stability tricks (e.g., log-sum-exp, epsilon in logs) .
- Construction and interpretation of **confusion matrices**, **precision/recall/F1**, and **macro averaging** for multiclass evaluation .
- Empirical comparison between **parametric (logistic regression)** and **non-parametric (KNN)** classifiers on the same dataset .

This makes the repository suitable both as a learning resource and as a project report for coursework in machine learning and pattern recognition .

## Author

This project was implemented by **Rezvan (2025)** as coursework and a learning exercise in machine learning and pattern recognition.
