import pandas as pd
import numpy as np
import datetime as dt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from pathlib import Path

prod = pd.read_csv("data/products_price.csv")

# Cleaning null price values

prod.price[prod.price < 0] = np.nan
prod = prod.dropna(subset=["price"])

# Converting to DateTime from Unix

prod["time"] = prod["time"].apply(dt.datetime.fromtimestamp)

# Categorizing color

prod.color[prod.color.isna()] = "None"
prod.color[prod.color == "_"] = "None"

colorCat = ["red", "orange", "yellow", "green", "blue", "purple", "violet", "gold", "silver", \
            "pink", "cyan", "brown", "black", "magenta", "white", "transparent", "navy", "olive", \
            "cream", "wine", "maroon", "beige", "gray", "multicolor"]

prod.color = prod.color.str.lower()

colorGroup = []
for col in prod.color:
    string = str(col)
    chosen = process.extractOne(string, colorCat, scorer=fuzz.partial_ratio)
    colorGroup.append(chosen[0])

prod['color'] = pd.Series(colorGroup)
prod.color[prod.color.isna()] = "Other"

# Dropping Size column

prod = prod.drop("size", axis=1)

# Creating 2D array of time and price

np.set_printoptions(suppress=True)

timePrice = prod[["time", "price"]]
A = timePrice.to_numpy()

prod['timePrice'] = A.tolist()

prodShort = prod.groupby(['asin', 'product_type', 'parent_asin', 'product_group', 'manufacturer', 'brand', \
                          'model', 'color', 'is_eligible_for_super_saver_shipping', 'is_sns'], group_keys=True, \
                         sort=False).timePrice.apply(list).reset_index()

# Output to csv

filepath = Path('CleanedData/testdata2.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
prodShort.to_csv(filepath)





