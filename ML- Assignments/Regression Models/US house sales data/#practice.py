#practice
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------
# STEP 1: Load Data
# ---------------------------------------------
df = pd.read_csv("us_house_Sales_data.csv")

# ---------------------------------------------
# STEP 2: Clean Numeric Columns
# ---------------------------------------------
df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
df['Bedrooms'] = df['Bedrooms'].str.extract('(\d+)').astype(float)
df['Bathrooms'] = df['Bathrooms'].str.extract('(\d+)').astype(float)
df['Area (Sqft)'] = df['Area (Sqft)'].str.replace(' sqft','', regex=False).astype(float)
df['Lot Size'] = df['Lot Size'].str.replace(' sqft','', regex=False).astype(float)

# ---------------------------------------------
# STEP 3: Drop Irrelevant Columns
# ---------------------------------------------
df = df.drop(['Address', 'MLS ID', 'Listing Agent','Listing URL'], axis=1)

# ---------------------------------------------
# STEP 4: One-Hot Encoding (Your Code)
# ---------------------------------------------
categorical_cols = ['City', 'State', 'Property Type', 'Status']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

ohe_array = ohe.fit_transform(df[categorical_cols])
print("OHE feature names:", ohe.get_feature_names_out(categorical_cols))

ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(categorical_cols))

# Drop original categorical columns
df = df.drop(categorical_cols, axis=1)

# Add encoded columns
df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

print("\n✅ Encoding Complete\n")
print(df.head())

# ---------------------------------------------
# STEP 5: Feature Matrix and Target
# ---------------------------------------------
X = df.drop('Price', axis=1)
y = df['Price']

# ---------------------------------------------
# STEP 6: Train-Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# STEP 7: Scaling Numerical Columns
# ---------------------------------------------
print(df.info())
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------
# STEP 8: Model Dictionary
# ---------------------------------------------
models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=450, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8),
    "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05),
    "SVR (RBF Kernel)": SVR(kernel='rbf'),
    "DecisionTree": DecisionTreeRegressor(max_depth=8)
}

# ---------------------------------------------
# STEP 9: Train & Evaluate Models
# ---------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n=== {name} ===")
    print(f"r2_score:  {r2:.4f}")
    print(f"mean_absolute_error:    {mae:.2f}")
    print(f"mean_squared_error: {mse:.2f}")
    print(f"root mean_squared_error: {rmse:.2f}")

print("\n✅ Model Training Complete")



For regression in Machine Learning, the goal is to analyze relationships between a continuous target variable and features, check linearity, detect outliers, and evaluate model performance.

Here are the most suitable and commonly used plots:

1. Scatter Plot

Used to see the relationship between feature (X) and target (y).

sns.scatterplot(x=df['feature'], y=df['target'])


Why: Shows trend direction (positive/negative), pattern, and outliers.

2. Pair Plot (or Scatter Matrix)

Visualizes all features against each other.

sns.pairplot(df)


Why: Good for identifying which features correlate with the target.

3. Correlation Heatmap

Shows strength of relationships.

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")


Why: Helps in feature selection.

4. Histogram / KDE Plot

Shows distribution (normal, skewed, bimodal, etc.)

sns.histplot(df['feature'], kde=True)


Why: Helps decide if scaling/log transform is needed.

5. Boxplot

Detects outliers.

sns.boxplot(x=df['feature'])


Why: Outliers may strongly affect regression.

6. Regression Line Plot

Shows line fitting and relationship.

sns.regplot(x='feature', y='target', data=df)


Why: Quickly shows if a linear relationship exists.

7. Residual Plot

Shows errors between predicted and actual values.

sns.residplot(x=y_pred, y=(y_test - y_pred))


Why: If residuals follow a random pattern → model is good.

8. Actual vs Predicted Plot

For model evaluation.

plt.scatter(y_test, y_pred)


Why: If points lie near diagonal line → good performance.

9. Learning Curve

Shows how performance changes with training size.

from sklearn.model_selection import learning_curve


Why: Helps detect overfitting / underfitting.

10. Error Distribution Plot

Distribution of prediction error.

sns.histplot(y_test - y_pred, kde=True)


Why: Tells if model systematically over/under predicts.

✅ Summary Table
Plot Name	Purpose
Scatter Plot	Analyze relationship between variable & target
Pair Plot	Feature relationships overview
Correlation Heatmap	Feature selection
Histogram / KDE	Check distribution / skewness
Boxplot	Detect outliers
Regression Plot	Fit line visualization
Residual Plot	Check model validity
Actual vs Predicted	Evaluate model accuracy
Learning Curve	Check over/under-fitting
Error Distribution	Check prediction bias