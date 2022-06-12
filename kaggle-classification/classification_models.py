# Ref: https://www.kaggle.com/code/atharvaingle/what-crop-to-grow

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import os

warnings.filterwarnings('ignore')


def Classification(model_name, data):
    Xtrain, Xtest, Ytrain, Ytest = data

    if model_name == "decisionTree":
        ClassModel = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
    elif model_name == "NaiveBayes":
        ClassModel = GaussianNB()
    elif model_name == "SVM":
        ClassModel = SVC(gamma='auto')
    elif model_name == "LogisticRegression":
        ClassModel = LogisticRegression(random_state=2)
    elif model_name == "RF":
        ClassModel = RandomForestClassifier(n_estimators=20, random_state=0)
    elif model_name == "XGBoost":
        ClassModel = xgb.XGBClassifier()
    else:
        raise ValueError(f"{model_name} is not valid!")

    ClassModel.fit(Xtrain, Ytrain)
    predicted_values = ClassModel.predict(Xtest)
    x = metrics.accuracy_score(Ytest, predicted_values)
    return ClassModel, predicted_values, x


# Save working directory
cd = os.path.dirname(os.path.abspath(__file__))
# Read dataset with pandas
df = pd.read_csv(os.path.join(cd, '../dataset/Crop_recommendation.csv'))

features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

sns.heatmap(df.corr(), annot=True)
plt.savefig(os.path.join(cd, "HeatmapData.png"))

# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []

# Splitting into train and test data


Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)
data = Xtrain, Xtest, Ytrain, Ytest

models = ["decisionTree", "NaiveBayes", "SVM", "LogisticRegression", "RF", "XGBoost"]

for m in models:
    class_model, predicted_values, accuracy = Classification(m, data)

    acc.append(accuracy)
    model.append(m)
    print(f"{m}'s Accuracy is: ", accuracy * 100)
    print(classification_report(Ytest, predicted_values))
    # Cross validation score (Decision Tree)
    score = cross_val_score(class_model, features, target, cv=5)
    print(score)

    # save classifier with pickle
    pkl_filename = f'models/{m}.pkl'
    # Adds working directory before file name: turns into absolute path for Permission Errors
    Model_pkl = open(os.path.join(cd+'/../', pkl_filename), 'wb')
    pickle.dump(class_model, Model_pkl)
    Model_pkl.close()

plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=acc, y=model, palette='dark')
plt.savefig(os.path.join(cd, "ComparisonClassificationModels.png"))

accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print(k, '-->', v)

"""data = np.array([[104, 18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)

data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction = RF.predict(data)
print(prediction)
"""
