import numpy as np
import pandas as pd
import getpass
import psycopg2
from psycopg2 import sql
import sshtunnel
import pickle
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


with open('unique_asins.pkl', 'rb') as f:
    unique_asins = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

data = pd.read_csv('data.csv')

# Load the trained model
model = load_model('best_model.h5')

# Assuming you have a connection to your PostgreSQL database
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
    
    # Create a new table for the predictions
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        asin VARCHAR(255),
        prediction_day INTEGER,
        predicted_price FLOAT,
        lower_bound FLOAT,
        upper_bound FLOAT
    )
    """
cur = conn.cursor()

# Generate the predictions for each ASIN
for asin in unique_asins:
    asin_data = data[data['asin'] == asin]
    asin_values = asin_data[columns].values.tolist()

    feature_vector = asin_values[-60:] if len(asin_values) >= 60 else [[0]*len(columns)] * (60 - len(asin_values)) + asin_values
    feature_vector = tf.convert_to_tensor([feature_vector], dtype=tf.float32)
    feature_vector = tf.reshape(feature_vector, (-1, 60, len(columns)))
    
    asin_predictions = []
    for _ in range(100):
        prediction = model.predict(feature_vector)[0]
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
        asin_predictions.append(prediction)
    
    asin_predictions = np.array(asin_predictions).reshape(100, 30)

    for prediction_day in range(30):
        # Select the predictions for this day and calculate the 90% prediction interval
        day_predictions = asin_predictions[:, prediction_day]
        lower_bound = np.percentile(day_predictions, 5)
        upper_bound = np.percentile(day_predictions, 95)
        predicted_price = np.mean(day_predictions)
        
        # Insert the prediction into the database
        insert_query = sql.SQL("""
        INSERT INTO predictions (asin, prediction_day, predicted_price, lower_bound, upper_bound)
        VALUES (%s, %s, %s, %s, %s)
        """)
        cur.execute(insert_query, (asin, prediction_day + 1, float(predicted_price), float(lower_bound), float(upper_bound)))

        # Commit the changes
        conn.commit()

        print(f"Inserted prediction for ASIN {asin}, day {prediction_day + 1}")

# Close the cursor and connection
cur.close()
conn.close()
