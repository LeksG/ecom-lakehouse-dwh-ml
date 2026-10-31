-- Фізична модель сховища даних (Gold) - схема «зірка» за Р. Кімбаллом
-- СКБД: PostgreSQL 16
DROP SCHEMA IF EXISTS dw CASCADE;
DROP SCHEMA IF EXISTS mart CASCADE;
DROP SCHEMA IF EXISTS feature_store CASCADE;
DROP SCHEMA IF EXISTS ml CASCADE;
CREATE SCHEMA dw;            -- ядро сховища: виміри та факти
CREATE SCHEMA mart;          -- дата-марти для звітності
CREATE SCHEMA feature_store; -- офлайн-сховище ознак для ML
CREATE SCHEMA ml;            -- результати інференсу моделей

CREATE TABLE dw.dim_date (
    date_key      INTEGER PRIMARY KEY,          -- YYYYMMDD
    full_date     DATE NOT NULL UNIQUE,
    day_of_month  SMALLINT NOT NULL,
    month         SMALLINT NOT NULL,
    month_name    VARCHAR(12) NOT NULL,
    quarter       SMALLINT NOT NULL,
    year          SMALLINT NOT NULL,
    day_of_week   SMALLINT NOT NULL,            -- 0 = понеділок
    is_weekend    BOOLEAN NOT NULL
);

CREATE TABLE dw.dim_customer (
    customer_key        SERIAL PRIMARY KEY,
    customer_unique_id  VARCHAR(32) NOT NULL UNIQUE,  -- природний ключ покупця
    city                VARCHAR(64),
    state               CHAR(2),
    zip_prefix          CHAR(5),
    first_purchase_date DATE
);

CREATE TABLE dw.dim_product (
    product_key      SERIAL PRIMARY KEY,
    product_id       VARCHAR(32) NOT NULL UNIQUE,
    category_pt      VARCHAR(64) NOT NULL,
    category_en      VARCHAR(64) NOT NULL,
    weight_g         NUMERIC(10,1),
    volume_cm3       NUMERIC(12,1),
    photos_qty       SMALLINT,
    description_len  INTEGER
);

CREATE TABLE dw.dim_seller (
    seller_key  SERIAL PRIMARY KEY,
    seller_id   VARCHAR(32) NOT NULL UNIQUE,
    city        VARCHAR(64),
    state       CHAR(2)
);

CREATE TABLE dw.dim_order_status (
    status_key   SMALLINT PRIMARY KEY,
    status_name  VARCHAR(16) NOT NULL UNIQUE,
    is_revenue   BOOLEAN NOT NULL               -- чи враховується у виручці
);

-- Факт продажів, зерно: одна позиція замовлення
CREATE TABLE dw.fact_sales (
    sales_key       BIGSERIAL PRIMARY KEY,
    order_id        VARCHAR(32) NOT NULL,        -- вироджений вимір
    order_item_id   SMALLINT NOT NULL,
    date_key        INTEGER  NOT NULL REFERENCES dw.dim_date(date_key),
    customer_key    INTEGER  NOT NULL REFERENCES dw.dim_customer(customer_key),
    product_key     INTEGER  NOT NULL REFERENCES dw.dim_product(product_key),
    seller_key      INTEGER  NOT NULL REFERENCES dw.dim_seller(seller_key),
    status_key      SMALLINT NOT NULL REFERENCES dw.dim_order_status(status_key),
    payment_type    VARCHAR(16),
    installments    SMALLINT,
    price           NUMERIC(10,2) NOT NULL,
    freight_value   NUMERIC(10,2) NOT NULL,
    item_total      NUMERIC(10,2) NOT NULL,
    delivery_days   NUMERIC(6,2),
    delay_days      NUMERIC(6,2),
    is_late         BOOLEAN,
    review_score    SMALLINT,
    UNIQUE (order_id, order_item_id)
);
CREATE INDEX ix_fact_sales_date     ON dw.fact_sales(date_key);
CREATE INDEX ix_fact_sales_customer ON dw.fact_sales(customer_key);
CREATE INDEX ix_fact_sales_product  ON dw.fact_sales(product_key);
