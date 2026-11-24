-- =============================================================================
-- Chương 4 — Chặng 1: Xây dựng Kho dữ liệu SQL Star Schema (Schema: dw)
-- Nguồn dữ liệu: Schema `raw` (tuyệt đối chỉ đọc, không chỉnh sửa dữ liệu gốc)
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS dw;

-- -----------------------------------------------------------------------------
-- 1. DIMENSION TABLES (5 Bảng Chiều)
-- -----------------------------------------------------------------------------

-- 1a. dw.dim_geolocation (Grain: 1 dòng = 1 mã bưu chính VARCHAR 5 ký tự)
CREATE OR REPLACE TABLE dw.dim_geolocation AS
SELECT
    geolocation_zip_code_prefix AS zip_code_prefix,
    AVG(geolocation_lat)        AS lat,
    AVG(geolocation_lng)        AS lng,
    MODE(geolocation_city)      AS city,
    MODE(geolocation_state)     AS state
FROM raw.geolocation
GROUP BY geolocation_zip_code_prefix;


-- 1b. dw.dim_customers (Grain: 1 dòng = 1 customer_id; giữ customer_unique_id)
CREATE OR REPLACE TABLE dw.dim_customers AS
SELECT
    c.customer_id,
    c.customer_unique_id,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state,
    g.lat,
    g.lng
FROM raw.customers AS c
LEFT JOIN dw.dim_geolocation AS g
    ON c.customer_zip_code_prefix = g.zip_code_prefix;


-- 1c. dw.dim_sellers (Grain: 1 dòng = 1 seller_id)
CREATE OR REPLACE TABLE dw.dim_sellers AS
SELECT
    s.seller_id,
    s.seller_zip_code_prefix,
    s.seller_city,
    s.seller_state,
    g.lat,
    g.lng
FROM raw.sellers AS s
LEFT JOIN dw.dim_geolocation AS g
    ON s.seller_zip_code_prefix = g.zip_code_prefix;


-- 1d. dw.dim_products (Grain: 1 dòng = 1 product_id)
-- Chuẩn hóa danh mục tiếng Anh, sửa tên cột sai chính tả, loại giá trị kích thước/trọng lượng <= 0
CREATE OR REPLACE TABLE dw.dim_products AS
SELECT
    p.product_id,
    p.product_category_name,
    COALESCE(
        t.product_category_name_english,
        p.product_category_name,
        'unknown'
    ) AS product_category_en,
    p.product_name_lenght        AS product_name_length,
    p.product_description_lenght AS product_description_length,
    p.product_photos_qty,
    CASE WHEN p.product_weight_g  > 0 THEN p.product_weight_g  END AS product_weight_g,
    CASE WHEN p.product_length_cm > 0 THEN p.product_length_cm END AS product_length_cm,
    CASE WHEN p.product_height_cm > 0 THEN p.product_height_cm END AS product_height_cm,
    CASE WHEN p.product_width_cm  > 0 THEN p.product_width_cm  END AS product_width_cm,
    CASE
        WHEN p.product_length_cm > 0
         AND p.product_height_cm > 0
         AND p.product_width_cm  > 0
        THEN p.product_length_cm * p.product_height_cm * p.product_width_cm
    END AS product_volume_cm3
FROM raw.products AS p
LEFT JOIN raw.category_translation AS t
    ON p.product_category_name = t.product_category_name;


-- 1e. dw.dim_date (Grain: 1 dòng = 1 ngày liên tục từ min đến max order_purchase_timestamp)
CREATE OR REPLACE TABLE dw.dim_date AS
WITH date_bounds AS (
    SELECT
        MIN(CAST(order_purchase_timestamp AS DATE)) AS min_date,
        MAX(CAST(order_purchase_timestamp AS DATE)) AS max_date
    FROM raw.orders
),
date_spine AS (
    SELECT
        UNNEST(generate_series(min_date, max_date, INTERVAL 1 DAY)) AS d
    FROM date_bounds
)
SELECT
    CAST(STRFTIME(d, '%Y%m%d') AS INTEGER) AS date_key,
    CAST(d AS DATE)                        AS full_date,
    EXTRACT(YEAR FROM d)::INTEGER          AS year,
    EXTRACT(QUARTER FROM d)::INTEGER       AS quarter,
    EXTRACT(MONTH FROM d)::INTEGER         AS month,
    STRFTIME(d, '%Y-%m')                   AS year_month,
    EXTRACT(WEEK FROM d)::INTEGER          AS week_of_year,
    EXTRACT(ISODOW FROM d)::INTEGER        AS day_of_week,
    DAYNAME(d)                             AS day_name,
    (EXTRACT(ISODOW FROM d) IN (6, 7))     AS is_weekend
FROM date_spine;


-- -----------------------------------------------------------------------------
-- 2. FACT TABLES (2 Bảng Sự kiện tách rõ Grain chống Fan-out)
-- -----------------------------------------------------------------------------

-- 2a. dw.fact_orders (Grain: 1 dòng = 1 đơn hàng, PK = order_id)
CREATE OR REPLACE TABLE dw.fact_orders AS
WITH items_agg AS (
    SELECT
        order_id,
        COUNT(*)                  AS item_count,
        SUM(price)                AS items_price_total,
        SUM(freight_value)        AS freight_total,
        SUM(price + freight_value) AS gross_order_value
    FROM raw.order_items
    GROUP BY order_id
),
ranked_payments AS (
    SELECT
        order_id,
        payment_type,
        payment_value,
        payment_installments,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY payment_value DESC NULLS LAST, payment_sequential ASC
        ) AS rn
    FROM raw.order_payments
),
payments_agg AS (
    SELECT
        order_id,
        SUM(payment_value)                        AS payment_total,
        COUNT(*)                                  AS payment_count,
        MAX(payment_installments)                 AS max_installments,
        MAX(CASE WHEN rn = 1 THEN payment_type END) AS primary_payment_type
    FROM ranked_payments
    GROUP BY order_id
),
ranked_reviews AS (
    SELECT
        order_id,
        review_id,
        review_score,
        review_creation_date,
        review_answer_timestamp,
        COUNT(*) OVER (PARTITION BY order_id) AS review_count,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY review_answer_timestamp DESC NULLS LAST,
                     review_creation_date DESC NULLS LAST,
                     review_id ASC
        ) AS rn
    FROM raw.order_reviews
),
reviews_dedup AS (
    SELECT
        order_id,
        review_id,
        review_score,
        review_creation_date,
        review_answer_timestamp,
        review_count
    FROM ranked_reviews
    WHERE rn = 1
)
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    CAST(o.order_purchase_timestamp AS DATE)              AS purchase_date,
    CAST(STRFTIME(o.order_purchase_timestamp, '%Y%m%d') AS INTEGER) AS purchase_date_key,
    STRFTIME(o.order_purchase_timestamp, '%Y-%m')         AS purchase_year_month,
    COALESCE(i.item_count, 0)                             AS item_count,
    COALESCE(i.items_price_total, 0.0)                    AS items_price_total,
    COALESCE(i.freight_total, 0.0)                        AS freight_total,
    COALESCE(i.gross_order_value, 0.0)                    AS gross_order_value,
    COALESCE(p.payment_total, 0.0)                        AS payment_total,
    COALESCE(p.payment_count, 0)                          AS payment_count,
    p.max_installments,
    p.primary_payment_type,
    r.review_id,
    r.review_score,
    r.review_creation_date,
    r.review_answer_timestamp,
    COALESCE(r.review_count, 0)                           AS review_count,
    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
         AND o.order_purchase_timestamp IS NOT NULL
         AND o.order_delivered_customer_date >= o.order_purchase_timestamp
        THEN EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_purchase_timestamp)) / 86400.0
    END AS delivery_days,
    CASE
        WHEN o.order_estimated_delivery_date IS NOT NULL
         AND o.order_purchase_timestamp IS NOT NULL
        THEN EXTRACT(EPOCH FROM (o.order_estimated_delivery_date - o.order_purchase_timestamp)) / 86400.0
    END AS estimated_delivery_days,
    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
         AND o.order_estimated_delivery_date IS NOT NULL
        THEN DATEDIFF('day', o.order_estimated_delivery_date, o.order_delivered_customer_date)
    END AS delivery_delay_days,
    COALESCE(
        o.order_status = 'delivered'
        AND o.order_delivered_customer_date > o.order_estimated_delivery_date,
        FALSE
    ) AS is_late_delivery,
    (
        COALESCE(o.order_status = 'delivered' AND o.order_delivered_customer_date IS NULL, FALSE)
        OR COALESCE(o.order_delivered_customer_date < o.order_purchase_timestamp, FALSE)
        OR COALESCE(o.order_approved_at < o.order_purchase_timestamp, FALSE)
        OR COALESCE(o.order_delivered_carrier_date < o.order_approved_at, FALSE)
        OR COALESCE(o.order_delivered_customer_date < o.order_delivered_carrier_date, FALSE)
    ) AS has_time_anomaly,
    COALESCE(
        o.order_status = 'delivered'
        AND o.order_delivered_customer_date IS NOT NULL
        AND o.order_purchase_timestamp IS NOT NULL
        AND o.order_delivered_customer_date >= o.order_purchase_timestamp,
        FALSE
    ) AS is_valid_delivered,
    CASE
        WHEN r.review_score IN (1, 2) THEN TRUE
        WHEN r.review_score IN (3, 4, 5) THEN FALSE
        ELSE NULL
    END AS is_low_review
FROM raw.orders AS o
LEFT JOIN raw.customers AS c
    ON o.customer_id = c.customer_id
LEFT JOIN items_agg AS i
    ON o.order_id = i.order_id
LEFT JOIN payments_agg AS p
    ON o.order_id = p.order_id
LEFT JOIN reviews_dedup AS r
    ON o.order_id = r.order_id;


-- 2b. dw.fact_order_items (Grain: 1 dòng = 1 sản phẩm trong đơn, PK = (order_id, order_item_id))
CREATE OR REPLACE TABLE dw.fact_order_items AS
WITH price_iqr AS (
    SELECT
        QUANTILE_CONT(price, 0.25) AS q1,
        QUANTILE_CONT(price, 0.75) AS q3
    FROM raw.order_items
    WHERE price IS NOT NULL
)
SELECT
    oi.order_id,
    oi.order_item_id,
    oi.product_id,
    oi.seller_id,
    o.customer_id,
    c.customer_unique_id,
    o.order_status,
    CAST(o.order_purchase_timestamp AS DATE)              AS purchase_date,
    CAST(STRFTIME(o.order_purchase_timestamp, '%Y%m%d') AS INTEGER) AS purchase_date_key,
    oi.shipping_limit_date,
    oi.price,
    oi.freight_value,
    (oi.price + oi.freight_value)                         AS item_total_value,
    (oi.freight_value / NULLIF(oi.price, 0))              AS freight_ratio,
    COALESCE(
        oi.price > (stats.q3 + 1.5 * (stats.q3 - stats.q1)),
        FALSE
    ) AS is_price_iqr_outlier
FROM raw.order_items AS oi
LEFT JOIN raw.orders AS o
    ON oi.order_id = o.order_id
LEFT JOIN raw.customers AS c
    ON o.customer_id = c.customer_id
CROSS JOIN price_iqr AS stats;
