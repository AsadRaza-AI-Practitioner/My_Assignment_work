#Fish Market Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df_=pd.read_csv('Fish Market Dataset.csv',delimiter=',')
print(df_)
le=LabelEncoder()
df_['Species_encoded']=le.fit_transform(df_["Species"])
print('class label mapping',dict(zip(le.classes_,le.transform(le.classes_))))
print(df_[['Species','Species_encoded']].head())
df=df_.drop(['Species'],axis=1)
print(df)
print(df.describe())
print(df.isnull().sum())
print(df.head())
print(df.tail())
print(df.duplicated().sum())
print(df.info())
print(df.shape)

X=df.drop('Weight',axis=1)
y=df['Weight']
print(X)
print(y)

X_test,X_train,y_test,y_train=train_test_split(X,y,train_size=.2,random_state=30)
print(f"train size: {round(len(X_train)/len(X)*100)}% \n\
test size: {round(len(X_test) /len(X) *100)}%")
scaler=StandardScaler()
# scaler.fit(X_train)
# X_train_scaled=scaler.transform(X_train.values)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled)
print(X_test_scaled)


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

regressor=LinearRegression(positive=True)
Rdg=Ridge(alpha=1.0,solver='auto',positive=True)
Lss=Lasso(1.0,positive=True)
Enet=ElasticNet(alpha=1.0,l1_ratio=0.5,positive=True)
svr=SVR(kernel='linear')
Gpr=GaussianProcessRegressor()
DTR=DecisionTreeRegressor(max_depth=None)

Models={
    'Linear_regression' :regressor,
    'Ridge Regression' :Rdg,
    'Lasso Regression' :Lss,
    'Elastic Net' : Enet,
    'Support vector resgression': svr,
    'Gaussian Process Regre' : Gpr,
    'Decision Tree Regre' :DTR }

for name, model in Models.items():
    model.fit(X_train_scaled,y_train)
    print(f"{name} result:\n this model is trained",sep='\n\n')
#models prediction si calculated by loop iteration

L_reg_pred=regressor.predict(X_test_scaled)
R_reg_pred=Rdg.predict(X_test_scaled)
Las_reg_pred=Lss.predict(X_test_scaled)
Elas_net_pred=Enet.predict(X_test_scaled)
Svr_pred=svr.predict(X_test_scaled)
Gpr_pred= Gpr.predict(X_test_scaled)
DRT_pred=DTR.predict(X_test_scaled)

model_prediction = [
    ('Linear Regression', L_reg_pred),
    ('Ridge Regression', R_reg_pred),
    ('Lasso Regression', Las_reg_pred),
    ('Elastic Net', Elas_net_pred),
    ('SVR', Svr_pred),
    ('Gaussian Process', Gpr_pred),
    ('Decision Tree', DRT_pred) ] 

# Loop through each model and print its R² score separately
from sklearn.metrics import r2_score
for model_name, y_pred in model_prediction:
    score = r2_score(y_test, y_pred)
    print(f"Model: {model_name}")
    print(f"R² Score: {score}")


# for name, model in Models.items():
#     y_pred=model.predict(X_test_scaled)
#     print(f"{name} result:\n prediction is achived {y_pred}  ",sep='\n\n')

# from sklearn.metrics import r2_score

# for name, model in Models.items():
#     print(f"{name} result:r2 score  is \n {r2_score(y_test,y_pred)}",sep="\n\n")

# y_pred=regressor.predict(X_test_scaled)
                #same results were noticed while prediction (negative values in prediction)
# result1=pd.DataFrame({'actual':y_test,'predict':L_reg_pred})
# print(f"model:linear regrssion")
# print('actual vs predicted \n',(result1))

# result2=pd.DataFrame({'actual':y_test,'predict':R_reg_pred})
# print(f"model:R_reg_pred")
# print('actual vs predicted \n',(result2))

# result3=pd.DataFrame({'actual':y_test,'predict':Las_reg_pred})
# print(f"model:Las_reg_pred")
# print('actual vs predicted \n',(result3))

# result4=pd.DataFrame({'actual':y_test,'predict':Elas_net_pred})
# print(f"model:Elas_net_pred")
# print('actual vs predicted \n',(result4))

# result5=pd.DataFrame({'actual':y_test,'predict':Svr_pred})
# print(f"model:Svr_pred")
# print('actual vs predicted \n',(result5))

# result6=pd.DataFrame({'actual':y_test,'predict':Gpr_pred})
# print(f"model:Gpr_pred")
# print('actual vs predicted \n',(result6))

# result7=pd.DataFrame({'actual':y_test,'predict':DRT_pred})
# print(f"model:DRT_pred")
# print('actual vs predicted \n',(result7))

for model_name, y_pred in model_prediction:
    result=pd.DataFrame({'actual':y_test,'predict':y_pred})
    print(f"model:{model_name}")
    print('actual vs predicted \n',(result))

from sklearn.metrics import mean_absolute_error,mean_squared_error
for model_name, y_pred in model_prediction:
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    print(f"model:{model_name}")
    print(f'mean absolute error :{mae}')
    print(f'mean square error: {mse}')
    print(f'root mean squared error:{rmse}')

#another metod dig deep into prediction of model accuracy
#R2 method
for model_name, y_pred in model_prediction:
    actual_minus_predicted=sum((y_test - y_pred)**2)
    actual_minus_actual_mean=sum((y_test - y_test.mean())**2)
    r2=1-actual_minus_predicted/actual_minus_actual_mean
    print(f"model:{model_name}")
    print('R2 :',r2)

from sklearn.metrics import mean_gamma_deviance,mean_tweedie_deviance

for model_name, y_pred in model_prediction:
    mgd=mean_gamma_deviance(y_test,y_pred).__round__(2)
    print('mean gamma deviance is\n',mgd)
    #mean_tweedie_deviance
    mtd1=mean_tweedie_deviance(y_test,y_pred,power=0).__round__(2)
    print('at power=0 mean tweedie deviance= mean square error',mtd1)
    mtd2=mean_tweedie_deviance(y_test,y_pred,power=1).__round__(2)
    print('power=1 it is equivalent to mean_poisson_deviance',mtd2)
    mtd3=mean_tweedie_deviance(y_test,y_pred,power=2).__round__(2)
    print('power=1 it is equivalent to  mean gamma deviance',mtd3)
#as we know at power=0 mean tweedie deviance= mean square error which was claculated 2.99
#power=1 it is equivalent to mean_poisson_deviance.
#by increasing power The difference in errors decreases. Finally, by setting, power=2 and it is equal to mean gamma deviance




