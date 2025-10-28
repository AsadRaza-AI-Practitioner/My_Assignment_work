#Heart disease checking from different symptoms
import pandas as pd
df=pd.read_csv('Heart disease checking.csv',delimiter="," )
print(df)
check_null=df.isnull()
print(check_null)
print(df.shape)

print("df.head() : ", df.head())


print(" df.info() ", df.info())

print(df.columns.tolist())

print(df.isnull().sum())

print(df.nunique())

print("df.tail()" , df.tail())

print('data type is',df.dtypes)

print(" df.describe()  ", df.describe().T)

#if objects are present in data types in rows of any corresponding column then print
# print(df.describe(include='object'))
column_of_data_set=df.columns
print(column_of_data_set)
#will give us all unique values in that selected column:
print(df.thal.unique())

#returns the values and their frequency:from which missing values can be find out
print(df.value_counts(dropna=False))

#returns the values and their frequency in that seleced column
print(df.thal.value_counts(dropna=False))


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isna(),cmap = 'Greens')
plt.show()
# Features and target
X = df.drop('target', axis=1)
y = df['target']

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