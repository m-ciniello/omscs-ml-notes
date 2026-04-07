WITH active_users AS (
  SELECT up.deduped_user_id
  FROM `sdp-prd-shop-ml.mart.mart__shop_app__user_profile` up
  WHERE up.last_shop_app_activity.activity_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {{ lookback_days }} DAY)
    AND (up.last_shop_app_user_geo.country_code IN ('US') OR up.country_code IN ('US'))
    AND up.risk_level IN ('safe/not rated', 'safe', 'low_risk')
    AND NOT up.is_on_giveaways_blocklist
    AND NOT up.is_shopifolk_user
),

ranked_orders AS (
  SELECT
    o.deduped_user_id,
    o.order_id,
    o.gmv_usd,
    o.created_at,
    ROW_NUMBER() OVER (
      PARTITION BY o.deduped_user_id
      ORDER BY o.created_at, o.order_id
    ) AS order_seq,
    MIN(o.created_at) OVER (PARTITION BY o.deduped_user_id) AS first_order_at
  FROM `sdp-prd-shop-ml.mart.mart__shared_data__shop_attributed_order_facts` o
  INNER JOIN active_users au
    ON o.deduped_user_id = au.deduped_user_id
  WHERE o.deduped_user_id IS NOT NULL
    AND o.gmv_usd IS NOT NULL
    AND o.attribution_type IN ('native', 'referral')
    AND o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {{ order_lookback_days }} DAY)
)

SELECT
  deduped_user_id,
  SUM(CASE WHEN order_seq <= 1 THEN gmv_usd END) AS gmv_first_1_order,
  SUM(CASE WHEN order_seq <= 2 THEN gmv_usd END) AS gmv_first_2_orders,
  SUM(CASE WHEN order_seq <= 3 THEN gmv_usd END) AS gmv_first_3_orders,
  SUM(CASE WHEN order_seq <= 4 THEN gmv_usd END) AS gmv_first_4_orders,
  SUM(CASE WHEN order_seq <= 5 THEN gmv_usd END) AS gmv_first_5_orders,
  SUM(CASE WHEN created_at < TIMESTAMP_ADD(first_order_at, INTERVAL 30 DAY)  THEN gmv_usd END) AS gmv_first_30_days,
  SUM(CASE WHEN created_at < TIMESTAMP_ADD(first_order_at, INTERVAL 90 DAY)  THEN gmv_usd END) AS gmv_first_90_days,
  SUM(CASE WHEN created_at < TIMESTAMP_ADD(first_order_at, INTERVAL 180 DAY) THEN gmv_usd END) AS gmv_first_180_days,
  MIN(created_at) AS first_order_at,
  COUNT(DISTINCT order_id) AS total_orders
FROM ranked_orders
GROUP BY deduped_user_id
