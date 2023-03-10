# Price Prediction Machine Learning Model
## Overview
This repository contains scripts and tools to facilitate a full stack implementation of a price prediction machine learning model.

This project was developed as part of the University of Washington Informatic's
2023 Capstone. The capstone team developing this project, Information 
Innovators, consists of the following:
- Grackie Jiang
- Franchezca Layog
- Victoria Murphy
- Samir Oudjani
- Harmeet Singh

For this current iteration of the project, the only supported retailer is
Amazon. Amazon historic price data for products is retrieved using the
[Keepa API](https://keepa.com/?#!api) which is a 3rd party service used
specifically to keep track of Amazon products.

## Repository Structure
This repository contains several subfolders that contains implementation for
specific aspects of the application. Each subfolder has a `README.md` that
provides basic information of its purpose.

These are the following subfolders:
- etl
    - Contains scripts to facilitate extract, load, and transform activities
    to retrieve information and load data into a warehouse for machine learning
    processing
- db
    - Contains SQL scripts to scaffold the database used to provide information
    used in the machine learning model
- ml
    - Contains scripts to interact with a database to pull historic price
    information and develop a machine learning model to predict future prices
