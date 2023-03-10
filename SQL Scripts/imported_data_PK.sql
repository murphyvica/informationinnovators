CREATE TABLE public.raw_prod_PK (
	RowID SERIAL PRIMARY KEY,
	asin VARCHAR (256),
	parent_asin VARCHAR (256),
    product_group VARCHAR (256),
    manufacturer VARCHAR (256),
    brand VARCHAR (256),
    model VARCHAR (256),
    color VARCHAR (256),
    is_eligible_for_super_saver_shipping boolean,
    is_sns boolean,
    "time" timestamp with time zone,
    price double precision
)

INSERT INTO raw_prod_pk ( 
	asin,
	parent_asin,
    product_group,
    manufacturer,
    brand,
    model,
    color,
    is_eligible_for_super_saver_shipping,
    is_sns,
    "time",
    price
)
SELECT asin,
    parent_asin,
    product_group,
    manufacturer,
    brand,
    model,
    color,
    is_eligible_for_super_saver_shipping,
    is_sns,
    "time",
    price
FROM raw_prod

SELECT * FROM raw_prod_pk