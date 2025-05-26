# Implementation Of Simple Machine Learning Algorithm to predict House Price
 1. Understand the Problem
You want to predict the price of a house based on various features like size, number of bedrooms, location, etc. This is a regression problem because the output (house price) is continuous.

2. Collect and Prepare the Data
Gather a dataset containing house features and their corresponding prices.

Common features might include:

Size in square feet

Number of bedrooms

Number of bathrooms

Location

Age of the house

Example data row:

Size (sqft)	Bedrooms	Bathrooms	Age	Price
2000	3	2	10	500000

Data cleaning: Handle missing values, remove outliers, encode categorical variables.

3. Split the Data
Divide data into training and testing sets, e.g., 80% train and 20% test.

Training data is used to build the model.

Test data is used to evaluate the model's accuracy.

4. Choose a Simple Machine Learning Model
A common simple model for regression is Linear Regression.

Linear regression assumes a linear relationship between features and the target price.

5. Train the Model
Use the training data to teach the model the relationship between features and house price.

The model finds coefficients (weights) for each feature to minimize prediction error.

Example in Python using scikit-learn:

python

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# X: features, y: target (price)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
6. Make Predictions
Use the trained model to predict house prices on the test data.

python

y_pred = model.predict(X_test)
7. Evaluate the Model
Measure the model's performance using metrics like:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R-squared (R²) score

python

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
8. Use the Model
Once you're satisfied with the model’s performance, use it to predict prices on new, unseen data.

Summary
Prepare and clean your data.

Split the data into train and test sets.

Train a simple linear regression model.

Predict house prices on test data.

Evaluate model accuracy.

Use the model for future predictions.
