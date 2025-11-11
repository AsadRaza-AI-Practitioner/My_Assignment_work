#Boston Dataset in Sklearn for regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df= pd.read_csv('advertising.csv', delimiter=',')
print(df)
print('null values ', df.isnull().sum)
print('summary of data', df.describe().T)
print('checking nans values in data', df.Sales.hasnans)
print('unique values in data drame',df.columns.unique)
print("column has how many number of unique values", df.columns.nunique)
print(' count how many Every unique combination appears in row ', df.value_counts(dropna=False)) #Every unique combination appears exactly once (1 at the end).This means DataFrame has no duplicate rows.
print(df)
print('checking shape of our df ', df.shape)
print("df['Tv]", df['TV'])
print("df[sale]", df['Sales'])

X=df[['TV','Radio','Newspaper']]
y=df['Sales']
print("x values are of TV",X)
print("Values for sales corresponding to Tv ", y)
print(X.shape)
print(X)
print(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=14)

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X)
X_scaled=scaler.transform(X.values)

X_train_scaled=scaler.fit_transform(X_train.values)
print(X_train_scaled[0])
print(X_train_scaled)
X_test_scaled=scaler.transform(X_test.values)
print(X_test_scaled[2])
print(X_test_scaled)

print(X.shape)
#Linear Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
#support vector machine model
from sklearn.svm import SVR
#Gaussian process models
from sklearn.gaussian_process import GaussianProcessRegressor
#Decision tree model
from sklearn.tree import DecisionTreeRegressor


# instantiate the model (using the default parameters) and enclose it in dictionary.
regressor=LinearRegression()
Rdg=Ridge(alpha=1.0)
Lss=Lasso(1.0)
Enet=ElasticNet(alpha=1.0,l1_ratio=0.5)
svr=SVR(kernel='linear')
Gpr=GaussianProcessRegressor()
DTR=DecisionTreeRegressor(max_depth=None,random_state=70)

Models={
    'Logistic_regression' :regressor,
    'Ridge Regression' :Rdg,
    'Lasso Regression' :Lss,
    'Elastic Net' : Enet,
    'Support vector resgression': svr,
    'Gaussian Process Regre' : Gpr,
    'Decision Tree Regre' :DTR }
for name, model in Models.items():
    model.fit(X_train_scaled,y_train)
    print(f"{name} result:\n this model is trained",sep='\n\n')
    

# regressor.fit(X_train_scaled,y_train)
# #intercept
# print(regressor.intercept_)
# #slope
# print(regressor.coef_)
# features_names=X.columns
# model_coefficient=regressor.coef_
# coefficient_df=pd.DataFrame(data=model_coefficient,
#                               index=features_names,
#                               columns=['coefficient value'])
# print(coefficient_df)

#after tarining the model through fit next step is to predict from model 

for name, model in Models.items():
    y_pred=model.predict(X_test_scaled)
    print(f"{name} result:\n prediction is achived {y_pred}  ",sep='\n\n')

from sklearn.metrics import r2_score

for name, model in Models.items():
    print(f"{name} result:r2 score  is \n {r2_score(y_test,y_pred)}",sep="\n\n")

result=pd.DataFrame({'actual':y_test,'predict':y_pred})
print('actual vs predicted \n',result)

from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

print(f'mean absolute error :{mae:.2f}')
print(f'mean square error: {mse:.2f}')
print(f'root mean squared error:{rmse:.2f}')

#another metod dig deep into prediction of model accuracy
#R2 method
actual_minus_predicted=sum((y_test - y_pred)**2)
actual_minus_actual_mean=sum((y_test - y_test.mean())**2)
r2=1-actual_minus_predicted/actual_minus_actual_mean
print('R2 :',r2)

from sklearn.metrics import mean_gamma_deviance,mean_tweedie_deviance

mgd=mean_gamma_deviance(y_test,y_pred,sample_weight=None).__round__(2)
print('mean gamma deviance is\n',mgd)

mtd1=mean_tweedie_deviance(y_test,y_pred,power=0).__round__(2)
print(mtd1)

#as we know at power=0 mean tweedie deviance= mean square error which was claculated 2.99

#power=1 it is equivalent to mean_poisson_deviance.
mtd2=mean_tweedie_deviance(y_test,y_pred,power=1).__round__(2)
print(mtd2)

#by increasing power The difference in errors decreases. Finally, by setting, power=2 and it is equal to mean gamma deviance

mtd3=mean_tweedie_deviance(y_test,y_pred,power=2).__round__(2)
print(mtd3)


