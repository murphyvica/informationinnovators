import getpass
import psycopg2
import sshtunnel
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

# Gather user input for SSH and database credentials
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

    # Query to select actual price from fact_price table for the last 30 days
    actual_query = """
        SELECT dp.asin, fp.price, dd.day
        FROM fact_price fp
        JOIN dim_product dp ON fp.product_id = dp.product_id
        JOIN dim_date dd ON fp.date_id = dd.date_id
        WHERE dd.date_id >= (SELECT MAX(date_id) - 30 FROM dim_date)
    """
    actual_data = pd.read_sql_query(actual_query, conn)

    # Query to select the predicted price from predictions table for the last 30 days
    prediction_query = """
        SELECT asin, predicted_price, prediction_day
        FROM predictions
        WHERE prediction_day <= 30
    """
    predicted_data = pd.read_sql_query(prediction_query, conn)

    # Close the connection
    conn.close()

# Merge actual and predicted dataframes
merged_data = pd.merge(actual_data, predicted_data, on=['asin', 'day'])

# Calculate percent difference
merged_data['percent_difference'] = abs(merged_data['price'] - merged_data['predicted_price']) / merged_data['price'] * 100

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(merged_data['price'], merged_data['predicted_price'])
print(f"Mean Absolute Percentage Error for the last 30 days: {mape * 100}%")
