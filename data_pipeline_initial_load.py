import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from datetime import timezone
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from pathlib import Path
import hashlib

import getpass
import urllib.parse
import psycopg2
import sshtunnel
from sqlalchemy import create_engine
from sqlalchemy import text
import os

# assign directory
directory = 'data\products'

# iterate over files in that directory

files = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        files.append(f)

for f in files:

    pd.options.mode.chained_assignment = None

    prod = pd.read_csv(f)

    # Cleaning null price values

    prod.price[prod.price < 0] = np.nan
    prod = prod.dropna(subset=["price"])

    # Converting to DateTime from Unix

    prod["time"] = prod["time"].apply(lambda t : dt.fromtimestamp(t, timezone.utc))

    # Categorizing color

    prod.color[prod.color.isna()] = "None"
    prod.color[prod.color == "_"] = "None"

    colorCat = ["red", "orange", "yellow", "green", "blue", "purple", "violet", "gold", "silver", \
                "pink", "cyan", "brown", "black", "magenta", "white", "transparent", "navy", "olive", \
                "cream", "wine", "maroon", "beige", "gray", "multicolor"]

    prod.color = prod.color.astype(str)
    prod.color = prod.color.str.lower()

    colorGroup = []
    for col in prod.color:
        string = str(col)
        chosen = process.extractOne(string, colorCat, scorer=fuzz.partial_ratio)
        colorGroup.append(chosen[0])

    prod['color'] = pd.Series(colorGroup)
    prod.color[prod.color.isna()] = "Other"

    # Dropping Size AND COLOR column

    prod = prod.drop("size", axis=1)
    # prod = prod.drop("color", axis=1)

    # Populating all days

    all_prods = prod.asin.unique()
    totalPROD = prod
    for p in all_prods:
        rows_prod = prod[prod.asin == p]
        startTime = min(rows_prod.time)
        setTime = startTime.replace(hour=23, minute=0)
        newPROD = rows_prod
        while (setTime < dt.now(timezone.utc)):
            time_change = timedelta(hours=24)
            setTime += time_change
            ref = rows_prod[rows_prod.time == max(rows_prod.time[rows_prod.time < setTime])]
            ref.time = setTime
            newPROD = pd.concat([newPROD, ref])
        totalPROD = pd.concat([totalPROD, newPROD])

    totalPROD = totalPROD.sort_values(['asin', 'time'])
    finalPROD = pd.concat([totalPROD, prod]).drop_duplicates(keep=False)
    finalPROD = finalPROD.reset_index()
    # Creating 2D array of time and price

    # np.set_printoptions(suppress=True)

    # timePrice = prod[["time", "price"]]
    # A = timePrice.to_numpy()

    # prod['timePrice'] = A.tolist()

    # prodShort = prod.groupby(['asin', 'product_type', 'parent_asin', 'product_group', 'manufacturer', 'brand', \
                              # 'model', 'color', 'is_eligible_for_super_saver_shipping', 'is_sns'], group_keys=True, \
                             # sort=False).timePrice.apply(list).reset_index()

    # Adding Hash

    # prod['rowhash'] = prod.apply(lambda x: hashlib.md5(str(tuple(x)).encode('utf-8')).hexdigest(), axis = 1)

    # Output to Database

    # ssh_host = input('SSH Host: ')
    # ssh_user = input('SSH Username: ')
    # ssh_pass = getpass.getpass('SSH Password: ')
    # db_host = input('DB Host: ')

    ssh_host = '100.85.132.56'
    ssh_user = 'vic'
    ssh_pass = 'infoc@p'
    db_host = 'orion'

    with sshtunnel.open_tunnel((ssh_host, 22), ssh_username=ssh_user, ssh_password=ssh_pass,
                               remote_bind_address=(db_host, 5432)) as tunnel:

        # db_user = input('PGSql Username: ')
        # db_pass_raw = getpass.getpass('PGSql Password: ')

        db_user = 'vic'
        db_pass_raw = 'infoc@p'

        db_pass = urllib.parse.quote_plus(db_pass_raw)

        engine = create_engine('postgresql://' + db_user + ':' + db_pass + '@:'
                               + str(tunnel.local_bind_port) + '/' + 'capstone')

        # finalPROD.to_sql(name='raw_prod', con=engine, schema='public', if_exists='append', index=True)

        # conn = psycopg2.connect(database='capstone', user=db_user, password=db_pass_raw,
                                # port=tunnel.local_bind_port)

        print("Connection Successful to PostgreSQL")

        with engine.connect() as conn:
            res = conn.execute(text("SELECT * FROM raw_prod ORDER BY asin ASC LIMIT 1")).fetchall()

            for row in res:
                print(row)

        # query = 'SELECT 1'
        # cursor = conn.cursor()
        # cursor.execute(query)

        # print(cursor.fetchall())





