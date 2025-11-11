#California Housing Data Set for Regression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
#1.  Data Set Loading and Description
'read dataframe'
df=pd.read_csv('California Housing Data Set.csv')
print(df.head())
print(df.shape)

#2. Data Preaparation
    # Check Missing Value in each Column
print(df.isnull().sum())
    # Plot Missing Value Column 
Tot_beds=df.total_bedrooms.hist()
print(Tot_beds)
    # Check Duplicate Value in each Column
print(df.columns.duplicated())
    # Check Duplicate Row
print(df.duplicated())
    # Check Columns Types
print(df.dtypes)
    # Missing value Filling with Median
df.total_bedrooms.fillna(df.total_bedrooms.median(), inplace=True)
print(df.isna().sum())
    # Now Finally Check Missing value
print(df.isnull().any())
#3. Exploratory Data Analysis (EDA)
print(df.describe())
    # Histploting for density evaluation of each column
    # cols=['longitude', 'latitude','housing_median_age','tatal_rooms','total_bedrooms','population','households', 'median_income','median_house_value','ocean_proximity']
fig,ax=plt.subplots(5,2,figsize=(9,9))
sns.histplot(data=df, x='longitude', ax=ax[0,0], kde=True, stat="density")
sns.histplot(data=df, x='latitude', ax=ax[0,1], kde=True, stat="density")
sns.histplot(data=df, x='housing_median_age', ax=ax[1,0], kde=True, stat="density")
sns.histplot(data=df, x='total_rooms', ax=ax[1,1], kde=True, stat="density")
sns.histplot(data=df, x='total_bedrooms', ax=ax[2,0], kde=True, stat="density")
sns.histplot(data=df, x='population', ax=ax[2,1], kde=True, stat="density")
sns.histplot(data=df, x='households', ax=ax[3,0], kde=True, stat="density")
sns.histplot(data=df, x='median_income', ax=ax[3,1], kde=True, stat="density")
sns.histplot(data=df, x='median_house_value', ax=ax[4,0], kde=True, stat="density")
sns.histplot(data=df, x='ocean_proximity', ax=ax[4,1], kde=True, stat="density")
plt.tight_layout()
plt.show()
    # Comparison of Ocean_proximity with House Value
sns.catplot(data=df,x='ocean_proximity',y='median_house_value',kind='bar')
plt.title("ocean_proximity VS median_house_value'")
plt.show()

sns.scatterplot(data=df,x='longitude', y= 'latitude',hue='median_house_value', palette='Greens',alpha=1)
plt.title("Scattering of price with Area")
plt.show()
    # Correlation of data
sns.pairplot(data=df)
plt.title('Correlation')
plt.show()
#4. Data Preprocessing
df['Rooms_per_Household'] = df['total_rooms']  /df  ['households']
df['Bedrooms_per_Room'] = df['total_bedrooms']  /df  ['total_rooms']
df['Population_per_Household'] = df['population'] /df  ['households']
print(df.head())
print(df.shape)
    # Categorical Encoding: Using One-Hot Encoding on the ocean_proximity feature
categorical_cols = ['ocean_proximity']
ohe = OneHotEncoder(sparse_output=False)
ohe_array = ohe.fit_transform(df[categorical_cols])
print("OHE feature names:", ohe.get_feature_names_out(categorical_cols))

ohe_df = pd.DataFrame(
    ohe_array, columns=ohe.get_feature_names_out(categorical_cols))
    # Drop Categorical_cols
df = df.drop(categorical_cols, axis=1)
print(df.tail())
    # Create df By Reseting Row indexing
df = pd.concat([df.reset_index(drop=True), ohe_df], axis=1)
df = df.rename(columns={'ocean_proximity_<1H OCEAN': 'ocean_proximity_one_H OCEAN'})

print(df.columns.tolist())
print(df.dtypes)
print(df.isnull().sum())
print(df.tail())
print(df.shape)

    # Finding  outlier
plt.figure(figsize=(4,3))
sns.boxplot(data=df, orient='h')
plt.show()
    # Removing Outtlier or Replacement with Mean
Out_cols=['total_rooms','population' ]    
for col in Out_cols:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3 -Q1
    min_r=Q1-(1.5*IQR)
    max_r=Q3+(1.5*IQR)
    Mean_Value=df[col].mean()
    df[col]=np.where((df[col]<min_r)|(df[col]>max_r),Mean_Value,df[col])
    # Final Checking Outliers
plt.figure(figsize=(4,3))
sns.boxplot(data=df, orient='h')
plt.show()
#2. Model Training and Selection:
    # Split the data into training and testing sets.

X=df.drop('median_house_value',axis=1)
y=df['median_house_value']
print('Features Column:',X)
print('Target Values Column:',y)

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.9,random_state=42)
print(f"train size:{round(len(X_train)/len(X) *100)}% )\n\
test size: {round(len(X_test) / len (X) *100 )}%")
    # Feature Scaling:
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled)
print(X_test_scaled)

# instantiate the model and enclose it in dictionary.

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
    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    # Metrics calcultions
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


for name, model in Models.items():
    # Apply HoldOut Validation Apprach Train and Test Split
    Result=model.score(X_test_scaled,y_test)
    print(' HoldOut Validation result:',Result)
    # Apply  Cross_val_score 
    CrV_score = cross_val_score(model, X, y, cv=10,scoring='r2')
    print("Cross-Validation Scores:", CrV_score)
    print("Mean C_V Score:", np.mean(CrV_score))
    
