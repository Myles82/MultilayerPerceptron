import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from MLP import *

# Fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# Extract data as pandas DataFrame
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine features and target into one DataFrame for easy filtering
data = pd.concat([X, y], axis=1)

# Drop rows with NaN values
cleaned_data = data.dropna()

# Split the cleaned data back into features (X) and target (y)
X = cleaned_data.iloc[:, :-1].to_numpy()  # Convert to NumPy array
y = cleaned_data.iloc[:, -1].to_numpy().reshape(-1, 1)  # Reshape to (n_samples, 1)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Standardize (mean=0, std=1)

# Split into training and validation sets (80% train, 20% validation)
# train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)
train_x, temp_x, train_y, temp_y = train_test_split(X, y, test_size=0.3, random_state=42)
val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

# Print dataset information
print(f"Training samples: {train_x.shape[0]}")
print(f"Validation samples: {val_x.shape[0]}")
print(f"Feature shape: {train_x.shape[1]}")
# Define the network architecture
layer1 = Layer(fan_in=train_x.shape[1], fan_out=64, activation_function=Relu())
layer2 = Layer(fan_in=64, fan_out=32, activation_function=Relu())
layer3 = Layer(fan_in=32, fan_out=1, activation_function=Linear())  # Linear output for regression

# Initialize random weights and biases
for layer in [layer1, layer2, layer3]:
    layer.W = np.random.randn(layer.fan_out, layer.fan_in) * 0.1
    layer.b = np.zeros((layer.fan_out, 1))

# Create MLP model
mlp = MultilayerPerceptron([layer1, layer2, layer3])

# Train model using Squared Error loss
training_losses, validation_losses = mlp.train(
    train_x=train_x,
    train_y=train_y,
    val_x=val_x,
    val_y=val_y,
    loss_func=SquaredError(),
    learning_rate=0.01,
    batch_size=32,
    epochs=50
)


# Evaluate on test set
test_predictions = mlp.forward(test_x)  # Forward pass through the network
test_loss = SquaredError().loss(test_predictions, test_y)  # Compute loss
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)  # Sum of squared residuals
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)

# Example usage:
r2 = r2_score(test_y, test_predictions)  # Assuming mlp.predict() returns predictions
print(f"R² Score: {r2 * 100:.2f}/100")

print(f"predicted: {test_predictions}")
print(f"true: {test_y}")
# Print test results
print(f"Test Loss: {test_loss}")
plt.show()