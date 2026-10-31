-- Дата-марти (Gold) для звіту з продажів
CREATE TABLE mart.sales_daily AS
SELECT d.full_date,
       COUNT(DISTINCT f.order_id)                       AS orders,
       COUNT(*)                                         AS items,
       SUM(f.price)                                     AS revenue,
       SUM(f.freight_value)                             AS freight,
       ROUND(SUM(f.item_total) / COUNT(DISTINCT f.order_id), 2) AS avg_check,
       COUNT(DISTINCT f.customer_key)                   AS customers
FROM dw.fact_sales f
JOIN dw.dim_date d         ON d.date_key = f.date_key
JOIN dw.dim_order_status s ON s.status_key = f.status_key
WHERE s.is_revenue
GROUP BY d.full_date;

CREATE TABLE mart.sales_category_monthly AS
SELECT d.year, d.month, p.category_en,
       COUNT(DISTINCT f.order_id) AS orders, SUM(f.price) AS revenue,
       ROUND(AVG(f.review_score), 2) AS avg_review
FROM dw.fact_sales f
JOIN dw.dim_date d    ON d.date_key = f.date_key
JOIN dw.dim_product p ON p.product_key = f.product_key
JOIN dw.dim_order_status s ON s.status_key = f.status_key
WHERE s.is_revenue
GROUP BY d.year, d.month, p.category_en;

CREATE TABLE mart.sales_state AS
SELECT c.state, COUNT(DISTINCT f.order_id) AS orders, SUM(f.price) AS revenue,
       ROUND(AVG(f.delivery_days), 1) AS avg_delivery_days,
       ROUND(100.0 * AVG(f.is_late::int), 1) AS late_pct
FROM dw.fact_sales f
JOIN dw.dim_customer c ON c.customer_key = f.customer_key
JOIN dw.dim_order_status s ON s.status_key = f.status_key
WHERE s.is_revenue
GROUP BY c.state;
