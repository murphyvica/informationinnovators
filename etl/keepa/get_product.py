"""This module provides implementation to search for product information"""
import os

import requests

KEEPA_TIME_OFFSET = 21564000


def get_product(asin: str) -> dict:
    """Retrieves product information for an ASIN

    :param asin: Product ASIN to search for
    :return: Information about the product
    """
    req_params = {
        'key': os.getenv('KEEPA_API_KEY'),
        'domain': 1,
        'asin': asin
    }

    response = requests.post('https://api.keepa.com/product', params=req_params)

    if not response.ok:
        return None

    product = response.json()['products'][0]

    product_data = []
    price_data: list = product['csv'][0]

    while len(price_data) > 0:
        product_data.append({
            'asin': product['asin'],
            'product_type': product['productType'],
            'parent_asin': product['parentAsin'],
            'product_group': product['productGroup'],
            'manufacturer': product['manufacturer'],
            'brand': product['brand'],
            'model': product['model'],
            'color': product['color'],
            'size': product['size'],
            'is_eligible_for_super_saver_shipping': product['isEligibleForSuperSaverShipping'],
            'is_sns': product['isSNS'],
            'time': (price_data.pop(0) + KEEPA_TIME_OFFSET) * 60,
            'price': price_data.pop(0) / 100
        })

    return product_data
