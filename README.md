# Clasificación de Ingresos con Spark ML

Este proyecto implementa un modelo de clasificación binaria con PySpark ML para predecir si una persona gana más de 50K al año, utilizando datos demográficos y laborales.

## Configuración del entorno (Linux/zsh)

### Crear un entorno virtual con venv

```zsh
python3 -m venv venv
source venv/bin/activate
```

### Instalar dependencias**

```zsh
pip install -r requirements.txt
```

### Instalar Java (requerido para PySpark)**

```zsh
sudo apt-get install openjdk-11-jdk
java -version
```

## Estructura del proyecto


## Ejecución

1. Ejecuta el pipeline:

```zsh
python main.py
```
## Income Classification with Tabular Data: Research Summary

## Project Overview
This project explores binary classification of income levels using a small tabular dataset (2,000 rows) with features such as age, workclass, education, sex, and hours per week. The goal is to predict whether an individual's income exceeds $50K.

## Data Preparation
- **Encoding:** Categorical variables (sex, workclass, education, label) were encoded numerically.
- **Feature Engineering:**
	- Created interaction features: `workclass_education_interaction`, `age_sex_interaction`, and `hours_per_week_squared`.
	- Dropped redundant columns after feature creation.
- **Scaling:**
	- Features were scaled by 10 and normalized to [0, 1] using MinMaxScaler.
- **Splitting:**
	- Dataset split: 10% test, 90% train+val; then 20% train, 80% val from train+val.

## Modeling Approaches
- **Logistic Regression:**
	- Used with class balancing and sample weights.
	- Evaluated with cross-validation and bootstrapping.
- **SVM (Support Vector Machine):**
	- RBF kernel, class balancing.
	- Evaluated with cross-validation and bootstrapping.
- **KNN and LightGBM:**
	- Compared on validation set.

## Model Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix.
- **Cross-Validation:**
	- Used StratifiedKFold to maintain class balance in splits.
	- Observed high variance in scores due to small dataset size.
- **Bootstrapping:**
	- Estimated mean and standard deviation of accuracy for SVM and Logistic Regression.

## Data Augmentation
- **Mixup:**
	- Attempted to generate new samples by linearly combining pairs of samples and labels.
	- Not effective for this dataset and models.
- **Other Techniques Discussed:**
	- SMOTE, oversampling, noise injection, feature engineering.

## Key Findings
- **Dataset Size:**
	- Small size (2,000 rows) is a major limitation, causing unstable and unreliable model performance.
- **Feature Engineering:**
	- Interaction features and polynomial features were explored to enrich the dataset.
- **Model Comparison:**
	- No model achieved strong generalization; performance was similar to random guessing.
- **Augmentation:**
	- Mixup did not improve results; other augmentation methods may be more suitable for larger or imbalanced datasets.

## Recommendations
- **Increase Dataset Size:**
	- More data is needed for reliable modeling and generalization.
- **Model Tuning:**
	- Tune hyperparameters and try ensemble methods.

## File Structure
- `eda.ipynb`: Main notebook with all code, analysis, and experiments.
- `README.md`: This summary of research, findings, and recommendations.

---
*This README documents all research, experiments, and findings from the notebook in this folder.*
