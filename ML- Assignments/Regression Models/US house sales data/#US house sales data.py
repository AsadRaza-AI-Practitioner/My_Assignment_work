#US house sales data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as wr
wr.filterwarnings('ignore')
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error


#Loading the Data Into DataFrame
df=pd.read_csv('us_house_Sales_data.csv',delimiter=',')
print(df)
# Checking the Types of data 
print(df.info())
print(df.shape)
print(df.head())
# Handling Missing Data
print(df.isnull().sum())
print(df.columns.duplicated())
print(df.describe().T)
# Dropping Irrelevent Columns
df.drop(['Address','Listing URL'],axis=1,inplace=True)

df['Price']=df['Price'].str.replace(",","")
df['Price']=df['Price'].str.replace("$","")
df['Price']=df['Price'].astype('Float64')
print(df['Price'].dtypes)
# Removing Unnecessary Signs/Symbols for dtype Conversion
df['Bathrooms']=df['Bathrooms'].str.replace(" ba","").astype('Float64')
df['Bedrooms']=df['Bedrooms'].str.replace(" bds","").astype('Float64')
df['Area (Sqft)']=df['Area (Sqft)'].str.replace(" sqft","").astype('Float64')
df['Lot Size']=df['Lot Size'].str.replace(" sqft","").astype('Float64')
print(df.info())
# Encoding Catagorial Variables
le=LabelEncoder()
df['City_encoded']=le.fit_transform(df['City'])
print('class label mapping',dict(zip(le.classes_,le.transform(le.classes_))))
df['State_encoded']=le.fit_transform(df['State'])
print('class label mapping',dict(zip(le.classes_,le.transform(le.classes_))))
df['Property Type_encoded']=le.fit_transform(df['Property Type'])
print('class label mapping',dict(zip(le.classes_,le.transform(le.classes_))))
df['MLS ID_encoded']=le.fit_transform(df['MLS ID'])
print('class label mapping',dict(zip(le.classes_,le.transform(le.classes_))))
df['Listing Agent_encoded']=le.fit_transform(df['Listing Agent'])
print('class label mapping',dict(zip(le.classes_,le.transform(le.classes_))))
df['Status_encoded']=le.fit_transform(df['Status'])
print('class label mapping',dict(zip(le.classes_,le.transform(le.classes_))))

df=df.drop(['City','State','Property Type','MLS ID','Listing Agent','Status','Zipcode','MLS ID_encoded'],axis=1)
print(df.info())
print(df.head())
print(df.nunique())
#Correlation bwtween Data
co_rel=df.corr()
print(co_rel)
# Correlation Through Ploting
Graph=sns.heatmap(co_rel,annot=True).set(title="Heat map of House sale Data")
plt.show()
# Outlier Detection & Handling depending on Algorithm
cols=['Price','Bedrooms','Bathrooms','Area (Sqft)','Lot Size','Year Built','Days on Market','City_encoded','State_encoded','Property Type_encoded','Listing Agent_encoded','Status_encoded']
plt.figure(figsize=(20,5))
sns.boxplot(data=df[cols], orient='h')
plt.show()

# If Outlier Exist in any Columns Can be Removed By:

# Q1=np.quantile(df['Bathrooms'],0.25)
# Q3=np.quantile(df['Bathrooms'],0.75)
# IQR=Q3 -Q1
# min_r=Q1-(1.5*IQR)
# max_r=Q3+(1.5*IQR)
# df=df[df['Bathrooms']<=max_r]

X=df.drop('Price',axis=1)
y=df['Price']
print(X)
print(y)
print(df.info())

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.9,random_state=42)
print(f"train size:{round(len(X_train)/len(X) *100)}% )\n\
test size: {round(len(X_test) / len (X) *100 )}%")


# scaler=StandardScaler()
# X_train_scaled=scaler.fit_transform(X_train)
# X_test_scaled=scaler.transform(X_test)
# print(X_train_scaled)
# print(X_test_scaled)



# instantiate the model (using the default parameters) and enclose it in dictionary.

Rf_regr=RandomForestRegressor( n_estimators=500,random_state=42)
Gb_regr=GradientBoostingRegressor(learning_rate=0.05)
Xgb_regr=XGBRegressor(n_estimators=500,random_state=42)
LGBM_regr=LGBMRegressor(num_leaves=31,random_state=42)
Sv_reg=SVR(kernel='rbf',gamma=0.1)
Dt_reg=DecisionTreeRegressor(max_depth=15,random_state=42)

Models={
    'RandomForestRegressor' :Rf_regr,
    'GradientBoostingRegressor' :Gb_regr,
    'XGBRegressor' :Xgb_regr,
    'LGBMRegressor' : LGBM_regr,
    'SVR': Sv_reg,
    'Decision Tree Regressor' :Dt_reg }
Resultant_Metrics=[]
for name, model in Models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    #metrics calcultion
    score = r2_score(y_test, y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    # Print results of metrics used.
    print(f"\n=== {name} ===")
    print(f"r2_score:  {score:.4f}")
    print(f"mean_absolute_error:    {mae:.4f}")
    print(f"mean_squared_error: {mse:.4f}")
    print(f"root mean_squared_error: {rmse:.4f}")

    # print(f"{name} result:\n this model is trained",sep='\n\n')
    #models prediction is calculated by loop iteration

# Rf_regr_pred=Rf_regr.predict(X_test_scaled)
# Gb_regr_pred=Gb_regr.predict(X_test_scaled)
# Xgb_regr_pred=Xgb_regr.predict(X_test_scaled)
# LGBM_regr_pred=LGBM_regr.predict(X_test_scaled)
# Sv_reg_pred= Sv_reg.predict(X_test_scaled)
# Dt_reg_pred=Dt_reg.predict(X_test_scaled)

# model_prediction = [
#     ('RandomForestRegressor', Rf_regr_pred),
#     ('GradientBoostingRegressor', Gb_regr_pred),
#     ('XGBRegressor', Xgb_regr_pred),
#     ('LGBMRegressor', LGBM_regr_pred),
#     ('SVR', Sv_reg_pred),
#     ('Decision Tree Regressor', Dt_reg_pred),] 

# Loop through each model and print its R² score separately
# for model_name, y_pred in model_prediction:
#     score = r2_score(y_test, y_pred)
#     print(f"Model: {model_name}")
#     print(f"R² Score: {score}")
    
# for model_name, y_pred in model_prediction:
#     result=pd.DataFrame({'actual':y_test,'predict':y_pred})
#     print(f"model:{model_name}")
#     print('actual vs predicted \n',(result))

# for model_name, y_pred in model_prediction:
#     mae=mean_absolute_error(y_test,y_pred)
#     mse=mean_squared_error(y_test,y_pred)
#     rmse=np.sqrt(mse)
#     print(f"model:{model_name}")
#     print(f'mean absolute error :{mae:.4f}')
#     print(f'mean square error: {mse:.4f}')
#     print(f'root mean squared error:{rmse:.4f}')
    



# #another metod dig deep into prediction of model accuracy
# #R2 method
# for model_name, y_pred in model_prediction:
#     actual_minus_predicted=sum((y_test - y_pred)**2)
#     actual_minus_actual_mean=sum((y_test - y_test.mean())**2)
#     r2=1-actual_minus_predicted/actual_minus_actual_mean
#     print(f"model:{model_name}")
#     print('R2 :',r2)

# print(X.shape)