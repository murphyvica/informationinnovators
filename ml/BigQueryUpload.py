from google.cloud import bigquery
import pickle
import numpy as np

predictions =''
n_simulations =''

with open('unique_asins.pkl', 'rb') as f:
    unique_asins = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)



# Create a BigQuery client
client = bigquery.Client()

# Set the dataset and table where you want to load the data
dataset_id = 'your_dataset_id'
table_id = 'your_table_id'

# Define your table schema
schema = [
    bigquery.SchemaField("asin", "STRING"),
    bigquery.SchemaField("prediction_day", "INTEGER"),
    bigquery.SchemaField("predicted_price", "FLOAT64"),
    bigquery.SchemaField("lower_bound", "FLOAT64"),
    bigquery.SchemaField("upper_bound", "FLOAT64"),
]

# Create a new table with the defined schema
table_ref = client.dataset(dataset_id).table(table_id)
table = bigquery.Table(table_ref, schema=schema)
table = client.create_table(table)

# Iterate over your predictions and insert them into the table
for asin, asin_predictions in zip(unique_asins, predictions):
    # Convert the predictions for this ASIN to a 2D numpy array
    asin_predictions = np.array(asin_predictions).reshape(n_simulations, 30)

    for prediction_day in range(30):
        # Select the predictions for this day and calculate the 90% prediction interval
        day_predictions = asin_predictions[:, prediction_day]
        lower_bound = np.percentile(day_predictions, 5)
        upper_bound = np.percentile(day_predictions, 95)
        predicted_price = np.mean(day_predictions)

        rows_to_insert = [
            {u"asin": asin, 
             u"prediction_day": prediction_day + 1, 
             u"predicted_price": predicted_price,
             u"lower_bound": lower_bound,
             u"upper_bound": upper_bound}
        ]

        # Insert rows into the table
        errors = client.insert_rows_json(table, rows_to_insert)

        # If there are any errors, raise an exception
        if errors:
            raise RuntimeError(f"Error inserting rows into BigQuery: {errors}")
