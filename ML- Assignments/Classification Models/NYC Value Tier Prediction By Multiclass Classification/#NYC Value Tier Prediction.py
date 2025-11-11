#NYC Value Tier Prediction
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as XGB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
  confusion_matrix, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score)


# 1.  Data Set Loading and Description
df=pd.read_csv('datasetupdated.csv')
print(df)
print(df.head())
print(df)
print(df.columns)
print(df.columns.tolist())
print(df.info())
print(df.describe().T)
print(df.isnull().sum())
print(df.dtypes)
print(df.shape)
print(df.nunique())
# 2.  Data Preprocessing (Creating Target and 15+ Features)
df['Total_Units']=df['RESIDENTIAL UNITS']+df['COMMERCIAL UNITS']
df['Age_At_Sale']=df['SALE YEAR']-df['YEAR BUILT']
print(df.columns.tolist())
      # Create Classification Target (Value_Tier) Using Sale Price
Q1 = df['SALE PRICE'].quantile(0.25)
Median = df['SALE PRICE'].quantile(0.50)
Q3 = df['SALE PRICE'].quantile(0.75)
      # Define the bins and labels
Bins = [df['SALE PRICE'].min(), Q1, Median, Q3, df['SALE PRICE'].max()]
Labels = [0, 1, 2, 3] 
      # Class 0 (Low), Class 1 (Medium), Class 2 (High), Class 3 (Luxury)

      # Create  'Value_Tier' the new target variable in Dataframe
df['Value_Tier'] = pd.cut(df['SALE PRICE'], bins=Bins, labels=Labels, include_lowest=True)

      # Check the class distribution (important for imbalanced data awareness)
print(df['Value_Tier'].value_counts())
print(df.nunique())
print(df['Value_Tier'])
print(df.head())
# 3.  Exploratory Data Anaysis
      # Value_Tier vs SALE PRICE Analysis by Visulization
sns.barplot(data=df,x='Value_Tier',y='SALE PRICE')
plt.title("Value_Tier vs SALE PRICE")
plt.show()
      # BOROUGH vs SALE PRICE Variance
sns.barplot(data=df,x='BOROUGH',y='SALE PRICE',orient='v')
plt.title("BOROUGH vs SALE PRICE")
plt.show()
      # By Caterplotting BOROUGH vs SALE PRICE Variance
sns.catplot(data=df,x='BOROUGH',y='SALE PRICE',orient='v')
plt.title("BOROUGH vs SALE PRICE")
plt.show()
      # By Barplotting BUILDING CLASS CATEGORY vs SALE PRICE
plt.figure(figsize=(4,3))
sns.barplot(data=df,x='BUILDING CLASS CATEGORY',y='SALE PRICE')
plt.title("BUILDING CLASS CATEGORY vs SALE PRICE")
plt.show()
      # Label Encoding
le=LabelEncoder()
df['BUILDING CLASS CATEGORY_encod']=le.fit_transform(df["BUILDING CLASS CATEGORY"])
print('class label mapping',dict(zip(le.classes_,le.transform(le.classes_))))
print(df[['BUILDING CLASS CATEGORY','BUILDING CLASS CATEGORY_encod']].head())
df=df.drop(['BUILDING CLASS CATEGORY'],axis=1)
      # One Hot Encoding to Catagorial Column
categorical_cols = ['BOROUGH', 'TAX CLASS AT PRESENT','NEIGHBORHOOD']
ohe = OneHotEncoder(sparse_output=False)
ohe_array = ohe.fit_transform(df[categorical_cols])
print("OHE feature names:", ohe.get_feature_names_out(categorical_cols))

ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(categorical_cols))
      # Drop Categorical_cols
df = df.drop(categorical_cols, axis=1)
print(df.tail())
      # Create df By Reseting Row indexing
df = pd.concat([df.reset_index(drop=True), ohe_df], axis=1)
print('dataframe now is',df )
print(df.columns.tolist())

      # Feature Scaling on Suggested Two Column in Suggestion provided
num_features=['GROSS SQUARE FEET','Age_At_Sale']
scaler=StandardScaler()
df[num_features]=scaler.fit_transform(df[num_features])
print(df.columns.tolist())
#4.   Model Training:
X=df.drop(['SALE PRICE','Value_Tier'],axis=1)
y=df['Value_Tier']
print('Features are :',X)
print('target are:',y)
print(df.dtypes)

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=42)
print(f"train size:{round(len(X_train)/len(X) *100)}% )\n\
test size: {round(len(X_test) / len (X) *100 )}%")
    # Feature Scaling:
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled)
print(X_test_scaled)
    # instantiate the model
Models={
    'logistic_regression' : LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGB.XGBClassifier( eval_metric='logloss', random_state=42)}

def Model_Evaluaton(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    Metrics = {
        'F1-Score': f1_score(y_test, y_pred,average='macro'),
        'Con_Mtx':confusion_matrix(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred,average='macro'),
        'Recall': recall_score(y_test, y_pred,average='macro'),
        'Cohen Kappa': cohen_kappa_score(y_test, y_pred)}
    return Metrics

Resultant_Metrics=[]
for name, model in Models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train_scaled,y_train)
    Metrics = Model_Evaluaton(model, X_test_scaled, y_test)
    Resultant_Metrics.append({
        'Model': name,
        'Metrics': Metrics
    })
    print("\nMetrics:")
    for i,j in Metrics.items():
        if isinstance(j, np.ndarray):
            print(f"{i}: {np.round(j, 4)}")
        else:
            print(f"{i}: {j:.4f}")
Resultant_df = pd.DataFrame(Resultant_Metrics)
print("\n\n===== Model Comparison =====")
print(Resultant_df)
# Save Resultant_df as Csv file
Resultant_df.to_csv('Resultant_df of resultant metrics.csv', index=False)
