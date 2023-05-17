Instructions for use: 

Predictionmodelcapstone.py: This is where the model is trained, it saves a checkpoint of each “best model” and from that model, and from that model you can write your predictions out to SQL. The model minimizes mean squared error loss, by looking at the previous 60 days of data (any longer and it begins to overfit) and some other categorical variables, and predicts the prices of products in the future. The main way to get the model to perform more accurately is to have the model train on a large number of epochs. This is adjustable within the code. I found the model produced the best predictions with between 85- 125 epochs for the dataset sizing we were using. The more epochs used, the closer to the current prices they will be, the fewer, the more likely to follow the trend line. 


SQLdump.py dumps the predictions into a postgresql database, and from there it can be uploaded to BigQuery using BigQueryUpload.py. You will have to make your own account and credentials though, as I have run out of free uploads to the API service. 



Please keep in mind that ASINS can be substituted for any unique product identifier:
This script (oredictionmodelcapstone.py) uses a recurrent neural network (RNN) with long short-term memory (LSTM) units to predict product prices based on various features such as the product's brand, category, model, color, size, and audience rating. The data is fetched from a PostgreSQL database through an SSH tunnel, preprocessed and then used to train the LSTM model.
Key steps and components of the code:
Import necessary libraries: Libraries necessary for creating SSH tunnels, connecting to PostgreSQL databases, data preprocessing, machine learning, and deep learning are imported.
Encode categorical features: The encode_categorical_features function is defined to convert categorical features into numeric format using the LabelEncoder from sklearn.preprocessing.
Establish SSH tunnel and connect to the database: The user is prompted for SSH and database credentials. An SSH tunnel is established and a connection is made to the PostgreSQL database.
Fetch and preprocess the data: A SQL query is executed to fetch the data. The data is preprocessed by converting categorical columns to numeric using the encode_categorical_features function, converting boolean columns to integer type, merging 'day', 'month', 'year' into a single datetime column, filling missing values, and ensuring that all inputs are numeric.
Filtering data: The script then filters out products (ASINs) that do not have sufficient data.
Saving preprocessed data and unique ASINs: The cleaned and preprocessed data is saved to a CSV file and the list of unique ASINs is saved using pickle for future use.
Create features and labels for model: The data for each unique ASIN is further processed into a format suitable for the LSTM model, with 60 days of data used as features to predict the next 30 days of prices.
Define and compile the model: An LSTM model is defined using the Sequential API from keras.models. The model is then compiled with the Adam optimizer and Mean Squared Error (MSE) as the loss function.
Train the model: The model is trained on the features and labels, with a ModelCheckpoint callback to save the best model based on validation loss.
Monte Carlo Dropout Predictions: The trained model is used to make predictions on the last 60 days of data for each ASIN using Monte Carlo Dropout. The 90% prediction interval is calculated for each ASIN and printed out. (this improved performance dramatically)


This script (SQLDump.py) is responsible for generating and storing predictions in a PostgreSQL database for each product (ASIN) using a previously trained model. The key steps are as follows:
Import necessary libraries: Libraries necessary for creating SSH tunnels, connecting to PostgreSQL databases, data preprocessing, machine learning, and deep learning are imported.
Load unique ASINs and feature columns: These were saved during the training phase and are loaded here for use in prediction.
Load the dataset and trained model: The preprocessed data is loaded from a CSV file, and the trained LSTM model is loaded using Keras' load_model function.
Establish SSH tunnel and connect to the database: The user is prompted for SSH and database credentials. An SSH tunnel is established and a connection is made to the PostgreSQL database.
Create a new table for the predictions: A new table named 'predictions' is created in the PostgreSQL database if it doesn't already exist. This table has columns for the ASIN, prediction day, predicted price, lower bound, and upper bound of the 90% prediction interval.
Generate and store the predictions: The script generates Monte Carlo Dropout predictions for the next 30 days for each unique ASIN, calculates the 90% prediction interval, and stores these values in the 'predictions' table in the PostgreSQL database. The mean of the predictions is used as the predicted price.
Commit and close the connection: After inserting the predictions for each ASIN, the changes are committed to the database and the cursor and connection are closed.

