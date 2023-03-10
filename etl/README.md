# Overview
This directory contains implementation for facilitating the ETL (extract, load, and transform) of data
into our OLAP (online  analytical processing) database. There are python modules to interact with
[Keepa's](https://www.keepa.com/#!api) API as well as python notebooks for data clean up and loading.

# Setup
A virtual environment can be easily created using `python -m venv venv` and you can install
dependencies using `pip install -r requirements.txt`. If you are developing on this, you may
want to also do `pip install -r dev-requirements.txt` to install linting and documentation tools.
