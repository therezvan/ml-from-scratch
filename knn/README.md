# ml-from-scratch
machine learning from scratch

# Breast Cancer Classification using Custom KNN (from Scratch)

This project implements a complete **K-Nearest Neighbors (KNN)** classification pipeline from scratch in Python to classify breast cancer data into malignant and benign classes. It compares three different neighbor search strategies (ordinary, heap-based, and KD-tree-based) in terms of accuracy and execution time, and visualizes both the data and model performance.

## Features

- Custom implementation of KNN (no use of scikit-learn’s KNN)  .
- Three neighbor search strategies:
  - Ordinary (brute-force) KNN
  - Heap-based KNN
  - KD-tree-based KNN  
- Manual implementation of:
  - Data preprocessing (cleaning, encoding, scaling, splitting)
  - K-fold cross-validation
  - KD-tree construction and query
  - Error calculation and basic evaluation metrics  
- Automatic selection of the best `k` using cross-validation  .
- Visualizations:
  - Feature histograms
  - Label distribution
  - Cross-validation error vs. `k`
  - Confusion matrix for the final model  .
- Runtime comparison between the three KNN variants  .

## Dataset

The project uses a breast cancer dataset (similar in structure to the UCI Breast Cancer Wisconsin dataset) stored in `Cancer_Data.csv`.

- Target column: `diagnosis`
  - `M` → malignant (mapped to `1`)
  - `B` → benign (mapped to `0`)  
- Irrelevant columns dropped:
  - `id`
  - `Unnamed: 32`  
- All remaining feature columns are standardized using `StandardScaler` to have zero mean and unit variance  .

## Project Structure

```
ml-from-scratch/
└── knn/
    │
    ├── main.py
    ├── src/
    │   ├── KNN.py
    │   ├── KDtree.py
    │   ├── preprocessor.py
    │   ├── evaluator.py
    │   └── visualizer.py
    ├── data/
    │   └── Cancer_Data.csv
    └── README.md
``
 

- `preprocessor.py`: Data loading, cleaning, normalization, and train–test split  .
- `KNN.py`: KNN classifier with ordinary, heap-based, and KD-tree-based prediction methods  .
- `KDtree.py`: KD-tree node and tree implementation plus `query` for k-nearest neighbors  .
- `evaluator.py`: K-fold cross-validation utility  .
- `visualizer.py`: Feature histograms, label distribution, and performance plots  .
- `main.py`: Entry point that ties everything together (preprocessing → model selection → evaluation → visualization)  .

## Methodology

1. **Preprocessing**
   - Load `Cancer_Data.csv`.
   - Drop `id` and `Unnamed: 32`.
   - Map `diagnosis` from `{M, B}` to `{1, 0}`.
   - Standardize all feature columns with `StandardScaler`.
   - Shuffle indices and split data into train and test sets (default 80/20, with `random_state=42`)  .

2. **K-fold Cross-Validation for k Selection**
   - Candidate values for `k`: `[1, 3, 5, 7, 9, 11, 13]`.
   - For each `k`, run 5-fold cross-validation using the custom `Evaluator`:
     - Train on 4 folds, validate on 1 fold (repeated for all folds).
     - Compute and print fold-wise accuracy and the average accuracy  .

   Cross-validation results:

   ```
   for k = 1
   Average Accuracy: 0.952

   for k = 3
   Average Accuracy: 0.961

   for k = 5
   Average Accuracy: 0.969

   for k = 7
   Average Accuracy: 0.962

   for k = 9
   Average Accuracy: 0.970

   for k = 11
   Average Accuracy: 0.965

   for k = 13
   Average Accuracy: 0.961

   Optimal k from cross-validation: 9
   ```
    

   The best performing value in cross-validation is **k = 9** with an average accuracy of **0.970**  .

3. **Training and Evaluation with the Optimal k**

   With `k = 9`:

   - The model is trained on the training set using the ordinary KNN implementation  .
   - Test performance is measured for three implementations:

   ```
   Ordinary: Test Error=0.0614, Time=0.3413 sec
   Heap:     Test Error=0.0614, Time=0.3790 sec
   KD-tree:  Test Error=0.0614, Time=0.1839 sec
   ```
    

   This corresponds to about **93.86% test accuracy** for all three variants, with the KD-tree implementation being the fastest on this dataset  .

4. **Visualization**

   The script generates several plots to better understand the data and model:

   - Feature histograms for the main input variables  .
   - Label distribution (number of malignant vs. benign samples)  .
   - Cross-validation error vs. `k` line plot (derived from the above accuracies)  .
   - Bar chart comparing execution time of:
     - Ordinary KNN
     - Heap-based KNN
     - KD-tree KNN  .
   - Confusion matrix for the final ordinary KNN classifier on the test set, displayed with a blue color map  .

## How to Run

1. Clone or copy the project into your environment (e.g., under `ml-from-scratch/knn`)  .
2. 2. Make sure the dataset is in the `data` folder:
3. Install dependencies:

   ```
   pip install numpy pandas matplotlib scikit-learn
   ```
    

4. Run the main script:

   ```
   python main.py
   ```
    

This will:

- Preprocess and split the dataset.
- Run 5-fold cross-validation for each candidate `k` and print the accuracies.
- Select the best `k` (here: 9) based on the highest average accuracy.
- Train and evaluate ordinary, heap-based, and KD-tree KNN on the test set.
- Print test errors and execution times for each method.
- Display plots including the confusion matrix and runtime comparison  .

## Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`  

## Notes and Possible Extensions

- Add distance-weighted KNN to give closer neighbors more influence on the prediction.
- Experiment with other distance metrics (e.g., Manhattan, Minkowski).
- Evaluate on other tabular classification datasets.
- Extend KD-tree code to support more advanced querying or benchmarking against library implementations  .

## Author

Developed by **Fateme Rezvan (2025)** as part of an educational project on implementing machine learning algorithms from scratch and analyzing the efficiency of different nearest neighbor search strategies  .
```
