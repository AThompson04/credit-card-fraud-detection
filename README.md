# Credit Card Fraud Detection 

## Dataset
The dataset for this project can be found on [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) under Database Contents License (DbCL) v1.0.

The authors of this dataset collected credit card transactions from two days in September 2013 by European cardholders. The dataset consists of 284 807 transactions of which 492 were fraudulent, making this dataset imbalanced as only 0.17% of transactions are fraudulent.

Due to confidentiality, the authors of the dataset could not provide the original features and additional background information. Therefore, the 28 of the 30 variables are the principal components obtained using PCA. The two variables *Time* and *Amount* were the only variables that were not transformed with PCA:
- **Time**: The seconds elapsed between the given transaction and the first transaction in the dataset. 
- **Amount**: he transaction amount.

## Objectives
The main objective of this project is:
> To develop a system that will be able to detect whether a transaction is fraudulent

To achieve this objective, it was further broken down into the following four technical sub-objectives:
1. To perform an exploratory data analysis
2. To clean the data where possible
3. To use oversampling methods to address the problem of imbalanced classes.
4. To develop supervised models to classify the nature of the card transaction on both the balanced and imbalanced data.

## Main Insights
Limited exploratory data analysis was done due to the nature of the dataset - V1 to V28 representing principal components obtained with PCA:
- The dataset is highly imbalanced, consisting of 283 253 fraudulent transactions and 473 non-fraudulent transactions.
- The distribution of the Time variable is bimodal. It is likely that the two peaks represent the transactions made during the day for the two day period. The distribution of Time for all non-fraudulent transaction was similar to the overall distribution, this was expected as most of the dataset is made up of non-fraudulent transactions. The distribution of Time for the fraudulent transactions does not follow the same distinct pattern for the whole dataset, and indicates that most fraudulent transactions took place 40 000 seconds after the first transaction.
- The amount spend on fraudulent transactions is significantly lower than on non-fraudulent transactions. The most spent on a fraudulent transaction was 2 125.87 with 50% of transaction being 9.82 or less, and the most spent on a non-fraudulent transaction was 25 691.16 with50% being 22 or less.
- There were slight positive and negative correlations between the principal components and the type of transaction. The correlations ranged from -0.3135 to 0.1490.

## Data Cleaning and Engineered Features
Due to the nature of the dataset no additional features were engineered, however there were 1 081 duplicate transactions. The duplicate transactions were removed from the dataset before model selection.

## Model Selection and Methodology
Due to the labelled nature of the dataset a supervised machine learning method was selected to predict whether a transaction was fraudulent or not. Different classification methods were explored to determine which method was best suited to the data and classifying the transactions.

The following classification methods were explored:
1. Logistic Regression
2. Decision Trees
3. Neural Networks

### Methodology:
The following methodology was used for all approaches:
- The dataset was split into a train, validation and test set using *Sklearn*'s *train_test_split*, with the split proportion being 70%, 15% and 15% respectively.
- To address the imbalanced classes in the dataset, the training dataset was resampled using the Synthetic Minority Oversampling Technique (SMOTE) and Adaptive Synthetic (ADASYN). These oversampling methods aim to balance the class distribution by synthesising new examples for the minority class. Oversampling was chosen as the resampling instead of under sampling because under sampling will remove instances of the majority class until each class has an equal number of observations - this would have resulted in 946 data points. However, there are factors to consider when oversampling for example the risk of overfitting - where the model learns characteristics of the replicated data too well. It is important to note that only the training dataset was resampled, as resampling the validation or test dataset could lead to misleading optimistic model evaluation and potential overfitting as the validation and test data should be representative of production data not the duplicated training data.

The original dataset and the two resampled datasets were used to train each of the classification methods.

The following methodology was used for the classification methods:
#### 1. Logistics Regression:
The following was done using the original and the two resampled datasets, resulting in three Logistic Regression models.
- The features that would be used for the Logistic Regression model were determined by: 
  - Using *Sklearn*'s *SelectFromModel* function for to determine the optimal features for the Logistic Regression model. The maximum number of features that were allowed to be selected ranged from one to all thirty features in the training dataset.
  - For each iteration of parameters a Logistic Regression model was trained using the selected features and the validation dataset was used to determine the prediction probabilities.
    - The predicted probabilities would then be used to find the optimal threshold. This was done by finding the threshold that produced the highest F1 score.
    - The F1 Score, RAC AUC Score and optimal threshold was recorded and printed for each observation.
  - The number of features for the option with the highest F1 Score and ROC AUC combination was used for the final Logistic Regression model.
- A Logistic Regression model was fit using the selected features using the training data set.
- The final predictions, the F1 Score, ROC AUC Score, ROC Curve and Confusion Matrix were calculated using the predictions from the test dataset and the optimal threshold for prediction that calculated when selecting the optimal number of features.

#### 2. Decision Trees:
The XGBoost algorithm was selected for model fitting due to its use of boosting, which helps address the bias-variance trade-off. Unlike other techniques such as bagging or ensembling, which focus on reducing high variance, boosting algorithms improve the model's accuracy by sequentially correcting errors from previous models. However, a known drawback of boosting is its susceptibility to overfitting.

To mitigate this risk and control model complexity, a grid search was performed during the fitting process to fine-tune hyper-parameters and prevent overfitting.

As with the Logistic Regression, Decision Trees were built for each of the three datasets.
- A grid search was performed testing the following parameters with the training dataset using *Sklearn*'s *GridSearchCV* and *xgboost*'s *XGBClassifier*:
  - The number of estimators [100, 200, 300]
  - The learning rate [0.1, 0.2, 0.3]
  - The maximum depth [3, 5, 10]
  - The subsample of data [0.5, 0.75, 1]
  - With 5-fold cross-validation and optimising using accuracy
- After determining the best hyper-parameters in the grid search, the optimal threshold for prediction was calculated by:
  - Calculating the prediction probabilities for the validation dataset.
  - Calculating the F1 Score for different thresholds.
  - The optimal threshold was determining by finding the threshold with the maximum F1 Score.
- The final predictions, the F1 Score, accuracy, ROC AUC Score, ROC Curve and Confusion Matrix were calculated using the predictions from the test dataset and the optimal threshold that was calculated using the validation dataset.

#### 3. Neural Networks:
- Sequential Neural Networks were trained for both the original and the balanced datasets using *tf_keras*.
- The models were designed for binary classification and consist of an input layer of 30 nodes, a hidden layer with 15 nodes, and an output layer with a single node. The input layer and the hidden layers used the Rectified Linear Unit (ReLU) activation function, followed by Dropout of 0.2 to prevent overfitting by randomly deactivating 20% of the nodes per layer during training. The Sigmoid activation was used on the output later to produce an output probability between 0 and 1.
- The model trained on the resampled dataset using SMOTE oversampling had two hidden layers with 15 nodes each. The hidden layers used the Rectified Linear Unit (ReLU) activation function, followed by Dropout of 0.2. Everything else remained the same as the previous point.
- The models were trained using the binary cross-entropy loss function and optimised using the Adam optimiser.
- To monitor generalisation, validation data was used to track performance. EarlyStopping was used to stop training if the validation loss did not improve for 25 consecutive epochs to avoid overfitting and unnecessary computation.
- The optimal threshold for prediction for each model was calculated by:
  - Calculating the prediction probabilities for the validation dataset.
  - Calculating the F1 Score for different thresholds.
  - The optimal threshold was determining by finding the threshold with the maximum F1 Score.
- The final predictions, the F1 Score, and the Confusion Matrix were calculated using the predictions from the test dataset and the optimal threshold that was calculated using the validation dataset.


### Model Selection:
In general the Decision Trees preformed the best out of all the classification methods for this set of data. The Logistic Regression models also preformed well but had lower F1 and ROC AUC scores than the Decision Trees. The neural networks performed significantly worse than the other classification methods, with low F1 scores and poor accuracy. Various combinations of hidden layers and nodes were tested, both with and without normalisation. However, due to time constraints and the superior performance of other methods, further tuning of the Neural Networks was discontinued.

See the table below for a summary of how each Decision Tree performed.


| Dataset | F1 Score | Accuracy | ROC AUC Score |
| :------: | :------: | :------: | :------: |
| Original Data | 0.8615 | 0.9996 | 0.9834 |
| Data using SMOTE Resampling | 0.8485 | 0.9996 | 0.9781 |
| Data using ADASYN Resampling | 0.8594 | 0.9996 | 0.9710 |

As seen in the table above the best performing Decision Tree was training on the original, as it scored slightly higher F1 Scores and ROC AUC Scores. It is beneficial that the model trained on the original dataset performed the best as the risk of overfitting due to oversampling are mitigated.

## Results and Metrics:
*Confusion matrix and ROC curve add in the threshold used, with a break down of:*

<figure><p align="center">
  <img src='/assets/ROC_CURVE_tree.png' style="width: 75%; height: 75%;">
  </br>
  <figcaption>
    Figure 1: ROC AUC Curve
  </figcaption></p></figure>
</br>
</br>

**Confusion Matrix:**

|| Non-Fraudulent | Fraudulent |
| :---: | :---: | :---: |
| **Non-Fraudulent** | 42 485 | 6 |
| **Fraudulent** | 12 | 56 |


- False Positives: 0.01412%
- Precision: 90.32258%
- Recall: 82.35294%
- F1 Score: 0.86154
- Accuracy: 99.9577%
