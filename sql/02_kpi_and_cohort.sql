-- =============================================================================
-- Chương 4 — Chặng 2: Xây dựng 4 bảng Data Mart KPI & Cohort Retention (Schema: dw)
-- Phạm vi nghiệp vụ chuẩn: Chỉ tính đơn hợp lệ (order_status NOT IN ('canceled', 'unavailable'))
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. dw.mart_monthly_kpi (Hiệu suất kinh doanh & vận hành theo tháng)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE dw.mart_monthly_kpi AS
WITH valid_orders AS (
    SELECT
        *,
        DATE_TRUNC('month', purchase_date)::DATE AS month_start_date
    FROM dw.fact_orders
    WHERE order_status NOT IN ('canceled', 'unavailable')
),
customer_first_month AS (
    SELECT
        customer_unique_id,
        MIN(month_start_date) AS first_month
    FROM valid_orders
    GROUP BY customer_unique_id
),
monthly_agg AS (
    SELECT
        v.purchase_year_month,
        v.month_start_date,
        COUNT(*) AS total_orders,
        COUNT(DISTINCT v.customer_unique_id) AS unique_customers,
        COUNT(
            DISTINCT CASE
                WHEN v.month_start_date = f.first_month THEN v.customer_unique_id
            END
        ) AS new_customers,
        COUNT(
            DISTINCT CASE
                WHEN v.month_start_date > f.first_month THEN v.customer_unique_id
            END
        ) AS returning_customers,
        SUM(v.payment_total)     AS gmv_payment,
        SUM(v.items_price_total) AS net_product_revenue,
        SUM(v.freight_total)     AS total_freight,
        AVG(CASE WHEN v.is_valid_delivered THEN v.delivery_days END) AS avg_delivery_days,
        AVG(
            CASE
                WHEN v.is_valid_delivered
                THEN CASE WHEN v.is_late_delivery THEN 100.0 ELSE 0.0 END
            END
        ) AS late_delivery_rate,
        AVG(v.review_score) AS avg_review_score,
        AVG(
            CASE
                WHEN v.is_low_review IS NOT NULL
                THEN CASE WHEN v.is_low_review THEN 100.0 ELSE 0.0 END
            END
        ) AS low_review_rate
    FROM valid_orders AS v
    INNER JOIN customer_first_month AS f
        ON v.customer_unique_id = f.customer_unique_id
    GROUP BY v.purchase_year_month, v.month_start_date
)
SELECT
    purchase_year_month,
    month_start_date,
    total_orders,
    unique_customers,
    new_customers,
    returning_customers,
    (returning_customers * 100.0 / NULLIF(unique_customers, 0)) AS repeat_customer_rate,
    gmv_payment,
    net_product_revenue,
    total_freight,
    (gmv_payment / NULLIF(total_orders, 0)) AS aov,
    avg_delivery_days,
    late_delivery_rate,
    avg_review_score,
    low_review_rate,
    (
        (gmv_payment - LAG(gmv_payment) OVER (ORDER BY month_start_date))
        * 100.0
        / NULLIF(LAG(gmv_payment) OVER (ORDER BY month_start_date), 0)
    ) AS mom_gmv_growth_pct
FROM monthly_agg
ORDER BY month_start_date;


-- -----------------------------------------------------------------------------
-- 2. dw.mart_cohort_retention (Ma trận giữ chân khách hàng theo tháng mua đầu tiên)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE dw.mart_cohort_retention AS
WITH valid_orders AS (
    SELECT
        customer_unique_id,
        DATE_TRUNC('month', purchase_date)::DATE AS order_month,
        payment_total
    FROM dw.fact_orders
    WHERE order_status NOT IN ('canceled', 'unavailable')
),
customer_cohort AS (
    SELECT
        customer_unique_id,
        MIN(order_month) AS cohort_month
    FROM valid_orders
    GROUP BY customer_unique_id
),
cohort_activity AS (
    SELECT
        c.cohort_month,
        DATEDIFF('month', c.cohort_month, v.order_month) AS cohort_index,
        COUNT(DISTINCT v.customer_unique_id)             AS active_customers,
        SUM(v.payment_total)                             AS cohort_gmv
    FROM valid_orders AS v
    INNER JOIN customer_cohort AS c
        ON v.customer_unique_id = c.customer_unique_id
    GROUP BY c.cohort_month, DATEDIFF('month', c.cohort_month, v.order_month)
),
cohort_with_size AS (
    SELECT
        cohort_month,
        STRFTIME(cohort_month, '%Y-%m') AS cohort_year_month,
        cohort_index,
        MAX(CASE WHEN cohort_index = 0 THEN active_customers END) OVER (
            PARTITION BY cohort_month
        ) AS cohort_size,
        active_customers,
        cohort_gmv
    FROM cohort_activity
)
SELECT
    cohort_month,
    cohort_year_month,
    cohort_index,
    cohort_size,
    active_customers,
    (active_customers * 100.0 / NULLIF(cohort_size, 0)) AS retention_rate_pct,
    cohort_gmv
FROM cohort_with_size
ORDER BY cohort_month, cohort_index;


-- -----------------------------------------------------------------------------
-- 3. dw.mart_category_performance (Hiệu suất theo ngành hàng, chống fan-out đơn hàng)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE dw.mart_category_performance AS
WITH valid_items AS (
    SELECT
        fi.order_id,
        COALESCE(dp.product_category_en, 'unknown') AS product_category_en,
        fi.price,
        fi.freight_value,
        fi.freight_ratio,
        fo.is_valid_delivered,
        fo.is_late_delivery,
        fo.review_score,
        fo.is_low_review
    FROM dw.fact_order_items AS fi
    INNER JOIN dw.fact_orders AS fo
        ON fi.order_id = fo.order_id
    LEFT JOIN dw.dim_products AS dp
        ON fi.product_id = dp.product_id
    WHERE fo.order_status NOT IN ('canceled', 'unavailable')
),
item_metrics AS (
    SELECT
        product_category_en,
        COUNT(*)           AS total_items_sold,
        SUM(price)         AS total_product_revenue,
        AVG(price)         AS avg_item_price,
        AVG(freight_value) AS avg_freight_value,
        AVG(freight_ratio) AS avg_freight_ratio
    FROM valid_items
    GROUP BY product_category_en
),
order_cat_distinct AS (
    SELECT DISTINCT
        order_id,
        product_category_en,
        is_valid_delivered,
        is_late_delivery,
        review_score,
        is_low_review
    FROM valid_items
),
order_metrics AS (
    SELECT
        product_category_en,
        COUNT(*) AS total_orders,
        AVG(
            CASE
                WHEN is_valid_delivered
                THEN CASE WHEN is_late_delivery THEN 100.0 ELSE 0.0 END
            END
        ) AS late_delivery_rate,
        AVG(review_score) AS avg_review_score,
        AVG(
            CASE
                WHEN is_low_review IS NOT NULL
                THEN CASE WHEN is_low_review THEN 100.0 ELSE 0.0 END
            END
        ) AS low_review_rate
    FROM order_cat_distinct
    GROUP BY product_category_en
)
SELECT
    im.product_category_en,
    om.total_orders,
    im.total_items_sold,
    im.total_product_revenue,
    im.avg_item_price,
    im.avg_freight_value,
    im.avg_freight_ratio,
    om.late_delivery_rate,
    om.avg_review_score,
    om.low_review_rate
FROM item_metrics AS im
INNER JOIN order_metrics AS om
    ON im.product_category_en = om.product_category_en
ORDER BY im.total_product_revenue DESC;


-- -----------------------------------------------------------------------------
-- 4. dw.mart_state_performance (Hiệu suất vận hành & hài lòng theo Bang — nền cho ANOVA)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE dw.mart_state_performance AS
SELECT
    customer_state,
    COUNT(*)                             AS total_orders,
    COUNT(DISTINCT customer_unique_id)   AS unique_customers,
    SUM(payment_total)                   AS total_gmv,
    (SUM(payment_total) / NULLIF(COUNT(*), 0)) AS aov,
    AVG(CASE WHEN is_valid_delivered THEN delivery_days END)    AS avg_delivery_days,
    MEDIAN(CASE WHEN is_valid_delivered THEN delivery_days END) AS median_delivery_days,
    AVG(
        CASE
            WHEN is_valid_delivered
            THEN CASE WHEN is_late_delivery THEN 100.0 ELSE 0.0 END
        END
    ) AS late_delivery_rate,
    AVG(review_score) AS avg_review_score,
    AVG(
        CASE
            WHEN is_low_review IS NOT NULL
            THEN CASE WHEN is_low_review THEN 100.0 ELSE 0.0 END
        END
    ) AS low_review_rate
FROM dw.fact_orders
WHERE order_status NOT IN ('canceled', 'unavailable')
GROUP BY customer_state
ORDER BY total_gmv DESC;
