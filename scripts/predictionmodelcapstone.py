import getpass
import numpy as np
import pandas as pd
import psycopg2
import tensorflow as tf
import sshtunnel
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from keras.models import Sequential
from keras.layers import Dense




ssh_host = input('SSH Host: ')
ssh_user = input('SSH Username: ')
ssh_pass = getpass.getpass('SSH Password: ')
db_host = input('DB Host: ')

with sshtunnel.open_tunnel((ssh_host, 22), ssh_username=ssh_user, ssh_password=ssh_pass,
                           remote_bind_address=(db_host, 5432)) as tunnel:

    db_user = input('PGSql Username: ')
    db_pass = getpass.getpass('PGSql Password: ')
    conn = psycopg2.connect(database='capstone', user=db_user, password=db_pass,
                            port=tunnel.local_bind_port)

    query = """
    SELECT dp.*, fp.price, dd.day, dd.month, dd.year
    FROM fact_price fp
    JOIN dim_product dp ON fp.product_id = dp.product_id
    JOIN dim_date dd ON fp.date_id = dd.date_id
    """    

    cursor = conn.cursor()
    cursor.execute(query)
    data = pd.read_sql_query(query, conn)
    data['price'] = data['price'].replace({'\$': '', ',': ''}, regex=True)

# Create a list of dictionaries to hold the features and labels for each ASIN
feature_dicts = []
results = []
label_dicts = []

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
df_features = df_features.astype(float)

features = tf.convert_to_tensor(df_features.values, dtype=tf.float32)
labels = tf.convert_to_tensor(df_labels.values, dtype=tf.float32)

# Define the model architecture
model = Sequential([])
model.add(Dense(128, input_shape=(143,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")

print(features.shape)
# Train the model
model.fit(features, labels, epochs=1, batch_size=32)

# Create a list to hold the predictions for each ASIN
predictions = []

# Iterate over each ASIN and make predictions
for asin in data['asin'].unique():
    asin_data = data[data['asin']==asin]
    manufacturer = asin_data['manufacturer'].iloc[0]
    brand = asin_data['brand'].iloc[0]
    asin_prices = asin_data['price'].tolist()
    actual_prices = asin_data['price'].tolist()
    actual_prices = np.array(actual_prices, dtype= float)


    
    # Create a feature vector for the ASIN
    feature_vector = asin_prices[-60:] + [manufacturer, brand]
    feature_vector = pd.get_dummies(pd.DataFrame([feature_vector], columns=[f"price_{i}" for i in range(60)] + ["manufacturer", "brand"]))
    feature_vector = tf.convert_to_tensor(feature_vector.values, dtype=tf.float32)

    # Make a prediction for the next 30 days using the trained model
    prediction = model.predict(feature_vector)[0]
    predictions.append(prediction)

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
    print(f"Actual Prices: {r['actual_prices']}")
    print(f"Predicted Prices: {r['predicted_prices']}")
    print(f"Accuracy: {r['accuracy']}")
    print(f"Confidence Interval: {r['confidence_interval']}")

# Print the predictions for each ASIN
for asin in data['asin'].unique():
    asin_data = data[data['asin']==asin]
    asin_features = asin_data['price'].tolist()[:60] + [asin_data['manufacturer'].iloc[0], asin_data['brand'].iloc[0]]
    asin_features = pd.get_dummies(pd.DataFrame(asin_features).T, columns=[60, 61])
    asin_features = asin_features.astype(float)
    asin_prediction = model.predict(asin_features)
    print(f"ASIN: {asin}")
    print(f"Next 30 day prices: {asin_prediction}")