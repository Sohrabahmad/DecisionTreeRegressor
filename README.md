# 🌳 Decision Tree Regression — Diabetes Dataset

## 📌 Project Overview

This project implements a **Decision Tree Regressor** using Scikit-learn's built-in Diabetes dataset to predict a quantitative disease-progression measure from patient-related features.

The model uses a decision tree with controlled depth and minimum leaf size to limit complexity. Its performance is evaluated using Mean Squared Error (MSE) and the R² score on training and testing data.

## 🎯 Objectives

- Load and explore the Diabetes dataset.
- Separate input features and the target variable.
- Split the dataset into training and testing sets.
- Train a Decision Tree Regression model.
- Evaluate model performance using MSE and R².
- Visualize the trained decision tree.

## 📂 Dataset

The notebook loads the dataset directly from Scikit-learn:

```python
from sklearn.datasets import load_diabetes

df = load_diabetes(as_frame=True).frame
```

The dataset contains **442 records and 11 columns**: 10 input features and one target variable.

### Features

| Feature | Description |
|---|---|
| `age` | Age-related feature |
| `sex` | Sex-related feature |
| `bmi` | Body mass index |
| `bp` | Average blood pressure |
| `s1` | Serum measurement 1 |
| `s2` | Serum measurement 2 |
| `s3` | Serum measurement 3 |
| `s4` | Serum measurement 4 |
| `s5` | Serum measurement 5 |
| `s6` | Serum measurement 6 |
| `target` | Quantitative measure of diabetes disease progression |

The feature values are standardized in the original Scikit-learn dataset. The target is a quantitative measure, making this a regression task.

## 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## ⚙️ Project Workflow

1. Import the required libraries.
2. Load the Diabetes dataset.
3. Inspect the dataset using `df.head()`.
4. Separate features (`X`) and target (`y`).
5. Split the data into training and testing sets.
6. Train a Decision Tree Regressor.
7. Generate predictions for training and testing data.
8. Evaluate the model using MSE and R².
9. Visualize the decision tree.

## 🤖 Model Implementation

The notebook uses the following configuration:

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth=7,
    min_samples_leaf=20
)

model.fit(X_train, y_train)
```

### Model Parameters

| Parameter | Value | Purpose |
|---|---:|---|
| `max_depth` | 7 | Limits the maximum depth of the tree |
| `min_samples_leaf` | 20 | Requires at least 20 training samples in each leaf |
| `test_size` | 0.3 | Reserves 30% of the dataset for testing |
| `random_state` | 42 | Makes the train-test split reproducible |

## 📊 Model Evaluation

The notebook evaluates the model using two regression metrics.

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values. Lower is better.
- **R² Score:** Measures how much target variation is explained by the model. Higher is generally better; a score of 1 indicates a perfect fit.

### Results

| Metric | Training Set | Testing Set |
|---|---:|---:|
| Mean Squared Error (MSE) | 2795.71 | 3177.66 |
| R² Score | 0.5453 | 0.4114 |

### Interpretation

- The training R² score is approximately **0.545**.
- The testing R² score is approximately **0.411**.
- The testing MSE is higher than the training MSE, indicating that prediction errors are larger on unseen data.
- The difference between training and testing performance suggests some generalization gap.

These results represent the model configuration used in the notebook, not necessarily the best possible performance for this dataset.

## 🌳 Decision Tree Visualization

The notebook visualizes the trained model using Scikit-learn's `plot_tree()` function.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.tight_layout()
plt.show()
```

This visualization helps illustrate the feature-based splits and prediction values learned by the tree.

## 📁 Project Structure

```text
Decision-Tree-Regression/
├── decision_tree_regressor.ipynb
└── README.md
```

The dataset is loaded directly from Scikit-learn, so a separate CSV file is not required.

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Decision-Tree-Regression
```

### 2. Install Dependencies

```bash
pip install pandas matplotlib scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the Notebook

Open `decision_tree_regressor.ipynb` and execute the cells sequentially.

An internet connection may be required the first time Scikit-learn downloads the dataset.

## 📚 Learning Outcomes

- Understanding regression problems
- Loading built-in Scikit-learn datasets
- Separating features and targets
- Train-test splitting
- Decision Tree Regression
- Controlling tree complexity
- Evaluating regression models using MSE and R²
- Visualizing decision trees

## ⚠️ Disclaimer

This project is intended for educational and machine learning practice purposes. The model's predictions should not be used for medical diagnosis or treatment decisions.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (AI & ML)
