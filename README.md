# Diabetes Prediction using SVM

## Project Overview

This project aims to build a predictive system using machine learning to determine whether a person has diabetes based on various diagnostic measurements. The system utilizes a Support Vector Machine (SVM) model trained on the PIMA Diabetes Dataset, a well-known dataset for this classification problem. The primary goal is to accurately predict the likelihood of diabetes in individuals, which could potentially aid in early detection and management.

## Dataset Description

The dataset used in this project is the PIMA Diabetes Dataset, sourced from Kaggle. This dataset is commonly used for binary classification problems to predict whether a patient has diabetes based on various diagnostic measurements.

The dataset contains the following features:

*   **Pregnancies:** Number of times pregnant
*   **Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test
*   **BloodPressure:** Diastolic blood pressure (mm Hg)
*   **SkinThickness:** Triceps skin fold thickness (mm)
*   **Insulin:** 2-Hour serum insulin (mu U/ml)
*   **BMI:** Body mass index (weight in kg/(height in m)^2)
*   **DiabetesPedigreeFunction:** Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
*   **Age:** Age (years)
*   **Outcome:** Class variable (0: non-diabetic, 1: diabetic)

### Loading and Exploring the Dataset

Below is the code to load the dataset using pandas and display its initial rows, shape, and descriptive statistics.

```python
# loading the diabetes dataset into a pandas dataframe
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# printing the first fives value of the dataset
print("First 5 rows of the dataset:")
display(diabetes_dataset.head())
```

```python
# numbers of the rows and columns in the dataset
print("\nShape of the dataset (rows, columns):")
print(diabetes_dataset.shape)
```

```python
# Getting the statistical measures of the data
print("\nDescriptive statistics of the dataset:")
display(diabetes_dataset.describe())
```

## Data Preprocessing

Before training the SVM model, the data undergoes several preprocessing steps to ensure it is in a suitable format.

### Separating Features and Labels

The first step is to separate the dataset into features (input variables, typically denoted as X) and the target variable (the outcome we want to predict, typically denoted as Y). In the PIMA Diabetes Dataset, the 'Outcome' column represents the target variable, while all other columns are considered features.

```python
# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
```

### Data Standardization

The features in the dataset have different ranges of values. To ensure that all features contribute equally to the model and to improve the performance of the SVM algorithm, the data is standardized. Standardization scales the data such that it has a mean of 0 and a standard deviation of 1. This is achieved using the `StandardScaler` from the `sklearn.preprocessing` module.

```python
scaler = StandardScaler()
```

```python
# Fit the scaler to the features and transform the data
scaler.fit(X)
standarized_data = scaler.transform(X)
```

```python
# Update X with the standardized data
X = standarized_data
```

The standardized features (`X`) and the labels (`Y`) are now ready for the next steps of model training and evaluation.

## Model Selection and Training

### Model Selection: Support Vector Machine (SVM)

For this binary classification problem (predicting diabetes or non-diabetes), a Support Vector Machine (SVM) classifier was chosen. SVMs are powerful supervised learning models used for classification and regression tasks. They work by finding the optimal hyperplane that best separates the data points of different classes in a high-dimensional space. SVMs are particularly effective in high-dimensional spaces and can use various kernel functions to handle non-linear relationships in the data, making them suitable for complex datasets like the PIMA Diabetes Dataset. A linear kernel is initially used for simplicity and interpretability.

### Splitting Data into Training and Testing Sets

To evaluate the model's performance on unseen data, the dataset is split into training and testing sets. A common split ratio is used, with 80% of the data allocated for training and 20% for testing. Stratified splitting is employed to ensure that the proportion of diabetic and non-diabetic outcomes is the same in both the training and testing sets, which is crucial for maintaining the representativeness of the dataset, especially with imbalanced classes. A random state is set for reproducibility.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

The shapes of the resulting sets are printed to confirm the split:

```python
print(X.shape, X_train.shape, X_test.shape)
```

### Training the SVM Classifier

The SVM classifier is trained using the training data (`X_train` and `Y_train`). The `svm.SVC` class from scikit-learn is used to create the classifier instance with a linear kernel. The `fit` method is then called to train the model.

```python
classifier = svm.SVC(kernel='linear')
```

```python
# Training the support vector machine classifier
classifier.fit(X_train, Y_train)
```

After training, the classifier is ready to make predictions on new data.

## Model Evaluation

After training the model, it is crucial to evaluate its performance to understand how well it generalizes to unseen data. Model evaluation helps in assessing the effectiveness of the trained model in making accurate predictions.

### Using Accuracy Score

The accuracy score is used as the evaluation metric for this classification model. Accuracy is defined as the ratio of correctly predicted instances to the total number of instances. It provides a measure of the overall correctness of the model's predictions. The accuracy is calculated for both the training data and the test data.

### Accuracy on Training Data

The accuracy score on the training data indicates how well the model has learned from the data it was trained on. A high training accuracy suggests that the model has captured the patterns in the training data.

```python
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```

```python
print('Accuracy score of the training data : ', training_data_accuracy)
```

### Accuracy on Test Data

The accuracy score on the test data is a more reliable measure of the model's performance on unseen data. It indicates how well the model is expected to perform in real-world scenarios. Comparing training accuracy and test accuracy can help identify issues like overfitting (where the model performs very well on training data but poorly on test data).

```python
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```

```python
print('Accuracy score of the test data : ', test_data_accuracy)
```

A higher accuracy score, for both training and test data, indicates a better performing model. However, it's important to look at both scores together to understand if the model is generalizing well or if there's a significant difference suggesting overfitting or underfitting.


## Making a Predictive System

Once the SVM model is trained, it can be used to predict the diabetes outcome for new, unseen data points. It is **critically important** that any new input data is preprocessed in the exact same way as the training data, especially by standardizing it using the *same* `StandardScaler` instance that was fitted on the original training data. This ensures that the new data has the same scale and distribution as the data the model was trained on.

Below are the steps and a code example demonstrating how to make a prediction for a single new instance:

1.  **Define the input data:** Represent the new data point as a tuple or list.
2.  **Convert to NumPy array:** Convert the input data into a NumPy array for numerical processing.
3.  **Reshape the array:** Reshape the array to `(1, n_features)` because the model expects input for a single instance in this format.
4.  **Standardize the data:** Use the *fitted* `scaler` object (the one used for the training data) to transform the reshaped input data.
5.  **Make the prediction:** Use the trained `classifier` object's `predict` method on the standardized input data.
6.  **Interpret the prediction:** The prediction will be `0` (Non Diabetic) or `1` (Diabetic).

```python
input_data = (10,125,70,26,115,31.1,0.205,41)

# change the input data into numpy array data
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Standardized the input data using the *fitted* scaler
std_data = scaler.transform(input_data_reshaped)
print('Standardized Input Data:', std_data)

# Make prediction
prediction = classifier.predict(std_data)
print('Prediction Result:', prediction)

if prediction[0] == 0:
  print('The person is predicted to be Non Diabetic')
else:
  print('The person is predicted to be Diabetic')
```

This predictive system can take new patient data as input and provide an automated prediction regarding their diabetes status based on the trained SVM model.

## Author

My name is Aditya Negi. I'm a developer with a strong interest in data science and machine learning, and I'm passionate about using data to build smart, impactful solutions.


## Summary:

### Data Analysis Key Findings

*   The project utilizes a Support Vector Machine (SVM) model for binary classification to predict diabetes based on the PIMA Diabetes Dataset.
*   The PIMA Diabetes Dataset from Kaggle contains 8 features and an 'Outcome' variable (0 for non-diabetic, 1 for diabetic).
*   Data preprocessing involved separating features (X) and labels (Y) and standardizing the features using `StandardScaler` to have a mean of 0 and a standard deviation of 1.
*   The dataset was split into training (80%) and testing (20%) sets using stratified sampling to maintain the proportion of outcomes in both sets.
*   A linear kernel SVM classifier was trained on the standardized training data.
*   Model evaluation was performed using the accuracy score, calculated for both the training data and the test data.
*   A predictive system was outlined, emphasizing the critical need to standardize new input data using the *same* fitted `StandardScaler` instance before making predictions with the trained model.
*   Relevant code snippets for data loading, exploration, preprocessing, model training, and prediction were included in the README content.
*   Suggestions for enhancing the README with visualizations such as histograms, scatter plots, correlation heatmap, confusion matrix, and an illustrative SVM decision boundary were provided.

### Insights or Next Steps

*   While accuracy provides an overall measure, consider including other metrics like precision, recall, and F1-score (possibly derived from a confusion matrix visualization) for a more comprehensive evaluation, especially given potential class imbalance in diabetes datasets.
*   Explore hyperparameter tuning for the SVM model (e.g., using GridSearchCV or RandomizedSearchCV) to potentially improve performance beyond the initial linear kernel model.
