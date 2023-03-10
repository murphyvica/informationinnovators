/* This SQL script contains implementation to scaffold a database using a dimensional modeling
schema (star schema) for historical price data of different products on Amazon.

SQL RDBMS: PostgreSQL 15 */

CREATE DATABASE capstone;

CREATE TABLE capstone.public.dim_date (
    date_id integer NOT NULL DEFAULT 'nextval('dim_date_date_id_seq'::regclass)',
    day integer,
    month integer,
    year integer,
    CONSTRAINT dim_date_pk PRIMARY KEY (date_id)
);

CREATE TABLE capstone.public.dim_product (
    product_id integer NOT NULL DEFAULT 'nextval('dim_product_product_id_seq'::regclass)',
    asin character varying(256) NOT NULL,
    parent_asin character varying(256),
    manufacturer character varying(256),
    brand character varying(256),
    category character varying(256),
    model character varying(256),
    color character varying(256),
    size character varying(256),
    audience_rating character varying(256),
    is_eligible_for_super_saving_shipping boolean,
    is_sns boolean,
    CONSTRAINT dim_product_pk PRIMARY KEY (product_id)
);

CREATE TABLE capstone.public.fact_price(
    product_id integer NOT NULL,
    price money NOT NULL,
    date_id integer NOT NULL,
    CONSTRAINT fact_price_pk PRIMARY KEY (product_id, date_id),
    CONSTRAINT fact_price_date_id_fk FOREIGN KEY (date_id)
        REFERENCES capstone.public.dim_date (date_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
        NOT VALID,
    CONSTRAINT fact_price_product_id_fk FOREIGN KEY (product_id)
        REFERENCES capstone.public.dim_product (product_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
        NOT VALID
);
