#! /bin/python3.9
"""
This module retrieves all immediate child categories of a given category. If no category
is specified, it will default to retrieving the child categories of the root category.
"""
import csv
import os
import sys
import time

import requests

if len(sys.argv) < 2:
    sys.exit('Please provide an output file name')


def get_category(category_id: int, writer: csv.DictWriter) -> None:
    """
    Recursively traverses the product tree and retrieves the category information
    for each category

    :param category_id: The ID of the category to traverse
    :param writer: Reference to a csv writer that will contain all output
    """

    query_parameters = {'key': os.environ['KEEPA_API_KEY'], 'domain': 1, 'category': category_id}
    response = requests.post('https://api.keepa.com/category', params=query_parameters, timeout=10)

    data = response.json()

    if response.status_code == 429:
        refill_milli = data['refillIn']
        time.sleep(refill_milli / 1000)

        response = requests.post('https://api.keepa.com/category',
                                 params=query_parameters, timeout=10)
        data = response.json()

    for category in data['categories']:
        writer.writerow(data['categories'][category])

        child_categories = data['categories'][category]['children']

        if child_categories is not None:
            for child_category in child_categories:
                get_category(child_category, writer)


with open(sys.argv[1], 'w', encoding='utf-8', newline='') as file:
    field_names = ['domainId', 'catId', 'name', 'parent', 'highestRank', 'productCount',
                   'contextFreeName', 'lowestRank']

    writer = csv.DictWriter(file, fieldnames=field_names, extrasaction='ignore')
    writer.writeheader()

    # Start at the root category 0
    get_category(0, writer)

sys.exit(0)
