import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Define the features and target variable
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
target = 'SalePrice'

# Preprocess the data
train = train[features + [target]].dropna()

# Split features and target
X = train[features]
y = np.log1p(train[target])  # Use log1p for log transformation

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate RMSE and R-squared for training and testing sets
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

# Output the performance metrics
print(f'Training RMSE: {train_rmse:.4f}')
print(f'Testing RMSE: {test_rmse:.4f}')
print(f'Test R-squared: {test_r2:.4f}')

# Prepare the test features from the test dataset
test_features = test[features]

# Predict the SalePrice for the test dataset
test_predictions = model.predict(test_features)

# Check if the length of test_predictions matches the length of the test DataFrame
if len(test_predictions) != len(test):
    raise ValueError(f"The length of test_predictions ({len(test_predictions)}) does not match the length of the test DataFrame ({len(test)})")

# If lengths match, proceed to create the submission DataFrame
test['SalePrice'] = np.expm1(test_predictions)  # Inverse of log1p
submission_df = pd.DataFrame({'Id': test['Id'], 'SalePrice': test['SalePrice']})

# Save the submission file
submission_df.to_csv('t1submission.csv', index=False)
