import getpass
import numpy as np
import pandas as pd
import psycopg2
import tensorflow as tf
import pickle
import sshtunnel
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def encode_categorical_features(df, cols):
    for col in cols:
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    return df

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
    WHERE dp.product_id IN (
    SELECT DISTINCT product_id
    FROM dim_product
    LIMIT 100
)
"""

    cursor = conn.cursor()
    cursor.execute(query)
    data = pd.read_sql_query(query, conn)
    data['price'] = data['price'].str.replace('[\$,]', '', regex=True).astype(float)



# Encode categorical features
categorical_cols = ['manufacturer', 'brand', 'category', 'model', 'color', 'size', 'audience_rating']
data = encode_categorical_features(data, categorical_cols)

# Convert boolean columns to integer type
boolean_cols = ['is_eligible_for_super_saving_shipping', 'is_sns']
for col in boolean_cols:
    data[col] = data[col].astype(int)

# Combine 'day', 'month', 'year' into a single datetime column and sort by this column
data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
data.sort_values(by='date', inplace=True, ascending=True)
data.drop(['year', 'month', 'day'], axis=1, inplace=True)
# Fill in missing values
data.fillna(-1, inplace=True)

# Convert float columns to int
for col in ['brand', 'model', 'size', 'audience_rating']:
    data[col] = data[col].astype(int)

# Check for sufficient data for each ASIN
asin_counts = data['asin'].value_counts()
sufficient_data_asins = asin_counts[asin_counts >= 90].index

# Only use ASINs with sufficient data
data = data[data['asin'].isin(sufficient_data_asins)]

print(f"Missing data:\n{data.isnull().sum()}")

# Check that all inputs are numeric
print(f"Data types:\n{data.dtypes}")

# Check for potential data scaling issues
print(f"Min price: {data['price'].min()}, Max price: {data['price'].max()}")

# Check for sufficient data for each ASIN
print(f"Number of days of data for each ASIN:\n{data['asin'].value_counts()}")

# Create features and labels
feature_dicts = []
label_dicts = []

# Select relevant columns
columns = ['price', 'manufacturer', 'brand', 'category', 'model', 'color', 'size', 'audience_rating', 'is_eligible_for_super_saving_shipping', 'is_sns']
with open('unique_asins.pkl', 'wb') as f:
    pickle.dump(data['asin'].unique(), f)


# Save columns
with open('columns.pkl', 'wb') as f:
    pickle.dump(columns, f)

# Save data to csv
data.to_csv('data.csv', index=False)

for asin in data['asin'].unique():
    asin_data = data[data['asin'] == asin]
    asin_values = asin_data[columns].values.tolist()
    
    for i in range(len(asin_values) - 90):
        feature_dicts.append({"features": asin_values[i:i + 60]})
        label_dicts.append({"labels": [x[0] for x in asin_values[i + 60:i + 90]]})  # Only take 'price' for labels

features = [d["features"] for d in feature_dicts]
labels = [d["labels"] for d in label_dicts]

# Convert to TensorFlow tensors
features = tf.convert_to_tensor(features, dtype=tf.float32)
labels = tf.convert_to_tensor(labels, dtype=tf.float32)

# Reshape the input data to be compatible with LSTM layers
features = tf.reshape(features, (-1, 60, len(columns)))

# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(60, len(columns)), activation='tanh', return_sequences=True, dropout=0.5))
model.add(LSTM(32, activation='tanh', dropout=0.5))
model.add(Dense(30))  # Predict the next 30 days prices

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

filepath = "best_model1.h5"

# Create a ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

# Train the model
model.fit(features, labels, epochs=97, batch_size=32, callbacks=[checkpoint], validation_split=0.2)

# Make predictions with Monte Carlo Dropout
n_simulations = 100
predictions = []
unique_asins = data['asin'].unique()  # Get only the first 10 unique ASINs

for asin in unique_asins:
    asin_data = data[data['asin'] == asin]
    asin_values = asin_data[columns].values.tolist()

    feature_vector = asin_values[-60:] if len(asin_values) >= 60 else [[0]*len(columns)] * (60 - len(asin_values)) + asin_values
    feature_vector = tf.convert_to_tensor([feature_vector], dtype=tf.float32)
    feature_vector = tf.reshape(feature_vector, (-1, 60, len(columns)))
    
    asin_predictions = []
    for _ in range(n_simulations):
        prediction = model.predict(feature_vector)[0]
        asin_predictions.append(prediction)
    
    predictions.append(asin_predictions)

    # Calculate and print the 90% prediction interval for each ASIN
    lower_bound = np.percentile(asin_predictions, 5)
    upper_bound = np.percentile(asin_predictions, 95)
    print(f"ASIN: {asin}")
    print(f"90% Prediction Interval: ({lower_bound}, {upper_bound})")



