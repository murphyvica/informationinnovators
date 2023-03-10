-----------------------------------
-- This script is run only after imported data is copied into table raw_prod_pk, using the 
-- "imported_data_PK" script
-----------------------------------
-- Product Table insert
-----------------------------------

INSERT INTO dim_product (asin, parent_asin, manufacturer, brand, category, model, color, is_eligible_for_super_saving_shipping, is_sns)
SELECT DISTINCT asin, parent_asin, product_group, manufacturer, brand, model, color, is_eligible_for_super_saver_shipping, is_sns
FROM raw_prod_pk

-----------------------------------
-- Date Table insert
-----------------------------------

INSERT INTO dim_date ("day", "month", "year")
SELECT DISTINCT EXTRACT('day' FROM "time"), EXTRACT('month' FROM "time"), EXTRACT('year' FROM "time")  FROM raw_prod_pk

-----------------------------------
-- Insert into fact_price
-----------------------------------
-- Get SPROCS:
-----------------------------------

CREATE OR REPLACE FUNCTION get_prodID (
IN asn VARCHAR (256),
IN clr VARCHAR (256),
OUT PID2 INT)
LANGUAGE plpgsql 
AS $$
BEGIN
	SELECT product_id INTO PID2 FROM dim_product WHERE asin = asn AND color = clr;
END;
$$;

CREATE OR REPLACE FUNCTION get_dateID (
IN dateP TIMESTAMP WITH TIME ZONE,
OUT DID2 INT)
LANGUAGE plpgsql 
AS $$
BEGIN
	SELECT date_id INTO DID2 FROM dim_date WHERE "day" = (EXTRACT('day' FROM dateP)) AND "month" = (EXTRACT('month' FROM dateP)) AND "year" = (EXTRACT('year' FROM dateP));
END;
$$;

-----------------------------------
-- Insert into fact_price SPROC:
-----------------------------------

CREATE OR REPLACE PROCEDURE insert_factprice (
asin VARCHAR (256),
col VARCHAR (256),
price MONEY,
dateT TIMESTAMP WITH TIME ZONE
)

LANGUAGE plpgsql
AS $$

DECLARE PID INT; DID INT;
BEGIN

PID := (SELECT * FROM get_prodID(asin, col));
DID := (SELECT * FROM get_dateID(dateT));

INSERT INTO public.fact_price(product_id, price, date_id)
VALUES(PID, price, DID);

COMMIT;
END;
$$;

-----------------------------------
-- WHILE LOOP INSERT:
-----------------------------------

do $$

DECLARE run INT; minPK INT; asn VARCHAR (256); clr VARCHAR (256); prc MONEY; dt TIMESTAMP WITH TIME ZONE; 

BEGIN
run := (SELECT COUNT(*) FROM raw_prod_pk);
WHILE run > 0 loop
	minPK := (SELECT MIN(rowid) FROM raw_prod_pk);
	
	SELECT asin, color, price::NUMERIC::MONEY, "time" INTO asn, clr, prc, dt
	FROM raw_prod_pk WHERE rowid = minPK;
	
	CALL insert_factprice(asn, clr, prc, dt);
	
	DELETE FROM raw_prod_pk WHERE rowid = minPK;
	run := run - 1;
end loop;
END;
$$;