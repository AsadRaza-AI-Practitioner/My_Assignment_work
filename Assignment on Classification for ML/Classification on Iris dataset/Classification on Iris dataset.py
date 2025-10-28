#Classification on Iris dataset
import pandas as pd
from sklearn.datasets import load_iris
Iris=load_iris(as_frame=True)
df=Iris.frame
print(df)
# Features and target
X = df.drop('target', axis=1)
y = df['target']
check_null=df.isnull()
print(check_null)
print(df.shape)

#if there is data is missing, it will display True else False.
print("x values are",X)
print("Yvalues are ",y)

# Take a preview
print("iris_df.head() : ", df.head())


print(" iris_df.info() ", df.info())

print(" iris_df.describe()  ", df.describe().T)
print(df.columns.tolist())
print(df.isnull().sum())
print(df.nunique())


print("iris_df.tail()" , df.tail())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                                  y,
                                                             train_size=.9,
                                                           random_state=25)

# Check the splits are correct
print(f"Train size: {round(len(X_train) / len(X) * 100)}% \n\
Test size: {round(len(X_test) / len(X) * 100)}%")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Instnatiating the models 
logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()
# Training the models 
logistic_regression.fit(X_train,y_train)
svm.fit(X_train,y_train)
tree.fit(X_train,y_train)

# Making predictions with each model
log_reg_preds = logistic_regression.predict(X_test)
svm_preds = svm.predict(X_test)
tree_preds = tree.predict(X_test)

from sklearn.metrics import classification_report

# Store model predictions in a dictionary
# this makes it's easier to iterate through each model
# and print the results. 
model_preds = {
    "Logistic Regression": log_reg_preds,
    "Support Vector Machine": svm_preds,
    "Decision Tree": tree_preds
}

#Display of each model result through for loop

for model, preds in model_preds.items():
    print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")
