#
import pandas as pd
df=pd.read_csv('predict+students.csv', delimiter=';')
print(df)
print("data head",df.head())
print("data info",df.info())
print("data describe",df.describe())
print("data frame tail",df.tail())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Target_encoded'] = le.fit_transform(df['Target'])
print("Class labels mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print(df[['Target', 'Target_encoded']].head())

from sklearn.preprocessing import StandardScaler
x=df.drop(['Target','Target_encoded'],axis=1)
y=df['Target_encoded']
print('x values are',x)
print(y)

#scaler = StandardScaler()
#scaler.fit(x)
# Transform features
#x_scaled = scaler.transform(x.values)

# View first instance
#print(x_scaled[0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,
                                                                  y,
                                                             train_size=.7,
                                                           random_state=25)

# Check the splits are correct
print(f"Train size: {round(len(X_train) / len(x) * 100)}% \n\
Test size: {round(len(X_test) / len(x) * 100)}%")

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
