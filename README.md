# Diabeties Prediction Machine Learning Model

## Project Overview
This project is a machine learning model designed to predict diabetes using the PIMA Diabetes Dataset. It involves data preprocessing, feature scaling, model training, and evaluation. The model is implemented using Python with libraries like NumPy, Pandas, and Scikit-Learn.

---

## Project Steps

### 1. Set Up the Environment
   - **Tools Used**: Visual Studio Code (VS Code) and Python
   - **Goal**: Create a structured workspace within VS Code and organize project folders for smooth development and data handling.

### 2. Download Diabetes Data
   - **Dataset Link**: [Diabetes Dataset]([https://www.kaggle.com/najir0123/walmart-10k-sales-datasets](https://www.dropbox.com/scl/fi/0uiujtei423te1q4kvrny/diabetes.csv?rlkey=20xvytca6xbio4vsowi2hdj8e&e=1&dl=0))
   - **Storage**: Save the data in the `data/` folder for easy reference and access.

### 3. Install Required Libraries and Load Data
   - **Libraries**: Install necessary Python libraries using:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```
   - **Loading Data**: Read the data into a Pandas DataFrame for initial analysis and transformations.
```
dd = pd.read_csv('diabetes.csv')
```

### 4. Explore the Data
   - **Goal**: Conduct an initial data exploration to understand data distribution, check column names, types, and identify potential issues.
   - **Analysis**: Use functions like `.info()`, `.describe()`, and `.head()` to get a quick overview of the data structure and statistics.
```
dd.head()
dd.shape
dd.describe()
dd['Outcome'].value_counts()
dd.groupby('Outcome').mean()
```

### 5. Data Cleaning & Preprocessing
```
## Seperating the outcome from the dataset
X = dd.drop(columns='Outcome', axis=1)
Y = dd['Outcome']

# Data Standardisation
scaler = StandardScaler()
scaler.fit(X)
standardised_data = scaler.transform(X)
X = standardised_data

## Train test split
## test size: the percentage of test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify   = Y, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)
```
 

### 6. Model Training
```
## Training the Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

### 7. Model Evaluation
```
## Model Evaluation
## Accuracy Score on the training data
X_train_prediction = classifier.predict(X_train) ## predictions of outcome of     diabeties
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) ## comparing prediction vs reality
print('Accuracy score of the training data: ', training_data_accuracy)

## Accuracy Score on the test data, data that the model has not seen/ trained with
X_test_prediction = classifier.predict(X_test) ## predictions of outcome of diabeties
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) ## comparing prediction vs reality
print('Accuracy score of the test data: ', test_data_accuracy)
```

### 8. Predicition System
```
## Making a Prediction System
input_data = (4,110,92,0,0,37.6,0.191,60) ## should be non-diabetic if correct

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshaping the array as we are predicting for 1 instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# input data standardisation
standardised_input = scaler.transform(input_data_reshaped)
print(standardised_input)

prediction = classifier.predict(standardised_input)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
```
### 9. Project Publishing and Documentation
   - **Documentation**: Maintain well-structured documentation of the entire process in Markdown or a Jupyter Notebook.
   - **Project Publishing**: Publish the completed project on GitHub or any other version control platform, including:
     - The `README.md` file (this document).
     - Jupyter Notebooks (if applicable).
     - Data files (if possible) or steps to access them.
---
## Requirements

- **Python 3.8+**
- **Libraries**:
  - `pandas`, `numpy`, `sqlalchemy`, `scikit-learn`

---

## Results and Insights

This section will include your analysis findings:
- **Model Accuracy**: The Support Vector Machine (SVM) model achieved an accuracy of 77% on the test dataset, indicating strong predictive performance in classifying diabetes cases.
- **Feature Importance**: Key factors influencing predictions included glucose levels, BMI, and age, suggesting their strong correlation with diabetes risk.
- **Impact of Feature Scaling**: Applying StandardScaler improved model stability, preventing bias from larger numerical features like glucose and insulin levels.
- **Data Distribution Trends**: Exploratory analysis showed that a higher proportion of diabetes cases were associated with higher glucose levels and BMI, aligning with medical expectations.
- **Potential Enhancement**: Additional improvements such as hyperparameter tuning, feature selection, and ensemble learning could further optimize prediction accuracy.

---

## Acknowledgments

- **Data Source**: Kaggle’s Diabetes Dataset
- **Inspiration**: @Siddhardhan
---
