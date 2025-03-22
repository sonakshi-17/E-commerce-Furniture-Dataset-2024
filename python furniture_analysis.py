import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Get the current directory and construct the file path
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'ecommerce_furniture_dataset.csv')

# Load dataset
df = pd.read_csv(file_path)

# Data Cleaning
print("Initial Missing Values:\n", df.isnull().sum())
df = df.dropna()  # Drop rows with missing values

# Convert 'tagText' to categories
df['tagText'] = df['tagText'].astype('category').cat.codes

# EDA - Distribution of 'sold' values
sns.histplot(df['sold'], kde=True)
plt.title('Distribution of Furniture Items Sold')
plt.show()

# EDA - Price vs Sales
sns.pairplot(df, vars=['originalPrice', 'price', 'sold'], kind='scatter')
plt.title('Price vs Sold Analysis')
plt.show()

# Feature Engineering
# Calculate discount percentage
df['discount_percentage'] = ((df['originalPrice'] - df['price']) / df['originalPrice']) * 100

# Convert productTitle to numeric features using TF-IDF
tfidf = TfidfVectorizer(max_features=100)
productTitle_tfidf = tfidf.fit_transform(df['productTitle'])
productTitle_tfidf_df = pd.DataFrame(productTitle_tfidf.toarray(), columns=tfidf.get_feature_names_out())
df = pd.concat([df, productTitle_tfidf_df], axis=1)
df = df.drop('productTitle', axis=1)

# Model Training
X = df.drop('sold', axis=1)
y = df['sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model Evaluation
# Linear Regression
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Results
print(f'Linear Regression MSE: {mse_lr:.2f}, R2: {r2_lr:.2f}')
print(f'Random Forest MSE: {mse_rf:.2f}, R2: {r2_rf:.2f}')

print("✅ Analysis Complete!")
