# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some example data (x: house size, y: house price)
np.random.seed(42)  # For reproducibility
x = 2 * np.random.rand(100, 1)  # House size (in 1000 square feet)
y = 4 + 3 * x + np.random.randn(100, 1)  # House price (in $1000s)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate and print the Mean Squared Error (MSE) to evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.scatter(x_test, y_test, color='blue', label='Actual data')
plt.plot(x_test, y_pred, color='red', label='Predicted line')
plt.xlabel('House Size (1000 sq ft)')
plt.ylabel('House Price ($1000s)')
plt.legend()
plt.show()



