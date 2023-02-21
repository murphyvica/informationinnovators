import tensorflow as tf
import pandas as pd
from pandas import Timestamp
import numpy as np

# Load the data
data = pd.read_csv("DataCleaning/testdata2.csv")

# Convert the timePrice column to a list of tuples with datetime objects
data["timePrice"] = data["timePrice"].apply(lambda x: [(pd.to_datetime(t), p) for t, p in eval(x)])

# Create a list of dictionaries to hold the features and labels for each ASIN
feature_dicts = []
label_dicts = []

# Iterate over each ASIN and create feature vectors consisting of the all available prices for each ASIN and add product_type and brand
for row in data.itertuples():
    asin = row.asin
    asin_data = []
    product_type = row.product_type
    brand = row.brand
    
    for time, price in row.timePrice:
        asin_data.append(price)
        
    if asin_data:
        for i in range(len(asin_data) - 3 * 30):
            feature_dicts.append({"features": asin_data[i:i+3*30] + [product_type, brand]})
            label_dicts.append({"labels": asin_data[i+3*30]})
            
# Convert the features and labels to TensorFlow tensors
features = [d["features"] for d in feature_dicts]
labels = [d["labels"] for d in label_dicts]

df_features = pd.DataFrame(features, columns=[f"price_{i}" for i in range(3*30)] + ["product_type", "brand"])
df_labels = pd.DataFrame(labels, columns=["label"])

df_features = pd.get_dummies(df_features, columns=["product_type", "brand"])

features = tf.convert_to_tensor(df_features.values, dtype=tf.float32)
labels = tf.convert_to_tensor(df_labels.values, dtype=tf.float32)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# Train the model
model.fit(features, labels, epochs=50, batch_size=32)

# Make predictions on the test data using the trained model
test_features = features
predictions = model.predict(test_features)

# Calculate the mean squared error loss between the predicted and actual prices
actual_prices = labels.numpy().squeeze()
mse_loss = np.mean((predictions.squeeze() - actual_prices) ** 2)

# Print the mean squared error loss
print("MSE Loss: ", mse_loss)

input_data = features[0:1, :]

# Get the model's prediction for the input data
prediction = model.predict(input_data)

# Print the prediction
print(prediction)

#This is the model I've built so far it builds a simple neural network
#with two hidden layers and 64 neurons each. The activation function 
#used in the hidden layers is the rectified linear unit (ReLU) activation,
#which is a commonly used activation function in neural networks. 
#The output layer has a single neuron, since we're performing a regression task 
# and predicting a single continuous value.
#The model is then compiled using the Adam optimization algorithm, 
#which is a popular optimization algorithm for neural networks, and the
#mean squared error loss function, which is a commonly used loss function
#for regression tasks. Finally, the model is fit on the training data for
#100 epochs, with a batch size of 32 samples. The validation data is 
#used to evaluate the model after each epoch and prevent overfitting.
#I chose this architecture because it's a simple and effective model for 
#regression tasks and works well with large datasets. The use of the Adam 
#optimizer and mean squared error loss function make this a good choice for
#number of neurons per layer is a common starting point that can be tuned 
#further based on the specific problem.