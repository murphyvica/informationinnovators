"""This module will index a list of products and then output them into a CSV file"""
import csv
import os
import sys

import requests

from get_product import get_product
from init_logger import logger

logger.info('Retrieving 200 products')

req_params = {
    'key': os.getenv('KEEPA_API_KEY'),
    'domain': 1
}

data = {
    'perPage': 200
}

response = requests.post('https://api.keepa.com/query', params=req_params, json=data)

if not response.ok:
    logger.error('Unable to complete request: %s', response.text)
    sys.exit('Exited because there was an error indexing products')

products = response.json()

with open('products.csv', 'w', newline='') as file:
    field_names = ['asin', 'product_type', 'parent_asin', 'product_group', 'manufacturer',
                   'brand', 'model', 'color', 'size', 'is_eligible_for_super_saver_shipping',
                   'is_sns', 'time', 'price']
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader()

    for asin in products['asinList']:
        product_info = get_product(asin)

        if product_info is None:
            logger.error('Unable to retrieve product information for %s', asin)
            sys.exit(1)

        for price_entry in product_info:
            writer.writerow(price_entry)

sys.exit(0)
