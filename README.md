# E-commerce-Furniture-Dataset-2024
🎯 1. Importing Libraries
python
Copy
Edit
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
✅ Explanation:

os: Helps manage file paths and directories.

pandas: Loads and manipulates data.

numpy: Supports numerical operations.

seaborn & matplotlib: Creates visualizations.

sklearn: Machine learning tools — handles text processing, splitting data, models, and evaluation.

📁 2. File Path Setup
python
Copy
Edit
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'ecommerce_furniture_dataset.csv')
✅ Explanation:
This makes sure the dataset loads correctly by constructing a path relative to where the script is saved — fixing the FileNotFoundError issue you had earlier.

🛠️ 3. Loading Data
python
Copy
Edit
data = pd.read_csv(file_path)
✅ Explanation:
Loads the CSV file into a DataFrame (a table-like structure).

🔍 4. Data Exploration
python
Copy
Edit
print(data.head())
print(data.isnull().sum())
✅ Explanation:

data.head() prints the first 5 rows for a quick overview.

data.isnull().sum() counts missing values per column.

🧹 5. Data Cleaning
python
Copy
Edit
data = data.dropna()
✅ Explanation:
Removes rows with missing data to avoid errors in analysis or model training.

🔬 6. Data Visualization (Your Output!)
python
Copy
Edit
sns.histplot(data['sold'], kde=True)
plt.title('Distribution of Furniture Items Sold')
plt.xlabel('sold')
plt.ylabel('Count')
plt.show()
✅ Explanation:

sns.histplot() creates a histogram of the sold column, showing how many items were sold and how often those numbers appear.

kde=True adds a smooth curve to visualize the trend.

plt.title(), xlabel(), and ylabel() set the title and axis labels.

Result: The graph you saw — most items sell in small numbers, and a few are top sellers. This is a typical pattern in sales data (a long-tail distribution).

🏗️ 7. Feature Engineering (Text Data Processing)
python
Copy
Edit
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(data['tagText'])
✅ Explanation:

TfidfVectorizer() converts text data (e.g., product tags) into numeric values — useful for models to "understand" the data.

fit_transform() prepares the text data for training.

📌 8. Defining Features & Target
python
Copy
Edit
X_numeric = data[['originalPrice', 'price']]
X = np.hstack((X_numeric, X_text.toarray()))
y = data['sold']
✅ Explanation:

X_numeric: Uses price columns as numeric features.

np.hstack() combines numeric and text features into one dataset.

y = data['sold']: Sets the number of items sold as the target (what we want to predict).

🔪 9. Split Data (Training vs. Testing)
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
✅ Explanation:
Splits data into:

80% for training the model.

20% for testing the model's performance.

🧠 10. Model Training (Linear Regression & Random Forest)
python
Copy
Edit
lr_model = LinearRegression()
rf_model = RandomForestRegressor()

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
✅ Explanation:

LinearRegression(): Fits a straight-line relationship between features and sales.

RandomForestRegressor(): Fits multiple decision trees and averages predictions for better performance.

🧾 11. Model Evaluation
python
Copy
Edit
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)

lr_r2 = r2_score(y_test, lr_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Linear Regression MSE: {lr_mse:.2f}, R2: {lr_r2:.2f}")
print(f"Random Forest MSE: {rf_mse:.2f}, R2: {rf_r2:.2f}")
✅ Explanation:

mean_squared_error(): Measures how far predictions are from actual sales. Lower is better.

r2_score(): Measures how well the model fits the data (1.0 is perfect, 0 means it’s guessing).

Result: You’ll see which model performs better.

✅ 12. Final Output
A visualization (histogram) of furniture sales distribution.

Performance metrics (MSE and R2) for both models.

