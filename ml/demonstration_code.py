import pandas as pd
import tensorflow as tf
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error

data = pd.read_csv("C:\\Users\\samir\\informationinnovators\\first_100.csv")
data['price'] = data['price'].replace({'\$': '', ',': ''}, regex=True)

# Create a list of dictionaries to hold the features and labels for each ASIN
feature_dicts = []
label_dicts = []
results = []
predictions =[]
# Iterate over each ASIN and create feature vectors consisting of the first 60 days of prices and add manufacturer and brand
for asin in data['asin'].unique():
    asin_data = data[data['asin']==asin]
    manufacturer = asin_data['manufacturer'].iloc[0]
    brand = asin_data['brand'].iloc[0]
    asin_prices = asin_data['price'].tolist()

    for i in range(len(asin_prices) - 90):
        feature_dicts.append({"features": asin_prices[i:i+60] + [manufacturer, brand]})
        label_dicts.append({"labels": asin_prices[i+60:i+90]})

# Convert the features and labels to TensorFlow tensors
features = [d["features"] for d in feature_dicts]
labels = [d["labels"] for d in label_dicts]

df_features = pd.DataFrame(features, columns=[f"price_{i}" for i in range(60)] + ["manufacturer", "brand"])
df_labels = pd.DataFrame(labels, columns=[f"price_{i}" for i in range(30)])

df_features = pd.get_dummies(df_features, columns=["manufacturer", "brand"])

features = tf.convert_to_tensor(df_features.values, dtype=tf.float32)
labels = tf.convert_to_tensor(df_labels.values, dtype=tf.float32)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(30)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# Train the model
model.fit(features, labels, epochs=50, batch_size=32)

# Make predictions on the test data using the trained model
test_features = features
prediction = model.predict(features)[0]
predictions.append(prediction)
actual_prices = asin_data['price'].tolist()
actual_prices = np.array(actual_prices, dtype=float)

# Print the mean squared error loss

input_data = features[0:1, :]
# Get the model's prediction for the next 30 days
mse = mean_squared_error(actual_prices[-30:].astype(float), prediction.astype(float))
accuracy = (mse / np.var(actual_prices[-30:]))
std_err = stats.sem(actual_prices[-30:])
conf_int = stats.t.interval(0.90, len(actual_prices[-30:])-1, loc=np.mean(actual_prices[-30:]), scale=std_err)
results.append({
        "asin": asin,
        "actual_prices": actual_prices[-30:],
        "predicted_prices": prediction,
        "accuracy": accuracy,
        "confidence_interval": conf_int
    })

# Print the results for each ASIN
for r in results:
    print(f"ASIN: {r['asin']}")
    print(f"Predicted Prices: {r['predicted_prices']}")
    print(f"Accuracy: {r['accuracy']}")
    print(f"Confidence Interval: {r['confidence_interval']}")