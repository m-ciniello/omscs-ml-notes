WITH eligible_profiles AS (
  SELECT up.deduped_user_id
  FROM `sdp-prd-shop-ml.mart.mart__shop_app__user_profile` up
  WHERE (up.last_shop_app_user_geo.country_code IN ('US') OR up.country_code IN ('US'))
    AND up.risk_level IN ('safe/not rated', 'safe', 'low_risk')
    AND NOT up.is_on_giveaways_blocklist
    AND NOT up.is_shopifolk_user
),

orders_sequenced AS (
  SELECT
    o.deduped_user_id,
    o.order_id,
    o.gmv_usd,
    o.created_at,
    ROW_NUMBER() OVER (
      PARTITION BY o.deduped_user_id
      ORDER BY o.created_at, o.order_id
    ) AS order_seq
  FROM `sdp-prd-shop-ml.mart.mart__shared_data__shop_attributed_order_facts` o
  INNER JOIN eligible_profiles ep
    ON o.deduped_user_id = ep.deduped_user_id
  WHERE o.deduped_user_id IS NOT NULL
    AND o.gmv_usd IS NOT NULL
    AND o.attribution_type IN ('native', 'referral')
),

sampled_users AS (
  SELECT DISTINCT deduped_user_id
  FROM orders_sequenced
  WHERE created_at <= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
  ORDER BY FARM_FINGERPRINT(CAST(deduped_user_id AS STRING))
  LIMIT {{ sample_size }}
),

sampled_orders AS (
  SELECT o.*
  FROM orders_sequenced o
  INNER JOIN sampled_users su
    ON o.deduped_user_id = su.deduped_user_id
),

milestone_anchors AS (
  SELECT
    deduped_user_id,
    order_seq AS milestone,
    created_at AS milestone_at
  FROM sampled_orders
  WHERE order_seq <= {{ max_milestone }}
    AND created_at <= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
),

post_milestone_gmv AS (
  SELECT
    a.deduped_user_id,
    a.milestone,
    a.milestone_at,
    COALESCE(SUM(o.gmv_usd), 0) AS gmv_365d_post,
    COUNT(DISTINCT o.order_id) AS orders_365d_post
  FROM milestone_anchors a
  LEFT JOIN sampled_orders o
    ON a.deduped_user_id = o.deduped_user_id
    AND o.created_at > a.milestone_at
    AND o.created_at <= TIMESTAMP_ADD(a.milestone_at, INTERVAL 365 DAY)
  GROUP BY a.deduped_user_id, a.milestone, a.milestone_at
),

next_order AS (
  SELECT
    a.deduped_user_id,
    a.milestone,
    MIN(o.created_at) AS next_order_at
  FROM milestone_anchors a
  INNER JOIN sampled_orders o
    ON a.deduped_user_id = o.deduped_user_id
    AND o.order_seq = a.milestone + 1
  GROUP BY a.deduped_user_id, a.milestone
)

SELECT
  p.deduped_user_id,
  p.milestone,
  p.milestone_at,
  CAST(p.gmv_365d_post AS FLOAT64) AS gmv_365d_post,
  p.orders_365d_post,
  n.next_order_at,
  TIMESTAMP_DIFF(n.next_order_at, p.milestone_at, DAY) AS days_to_next_order
FROM post_milestone_gmv p
LEFT JOIN next_order n
  ON p.deduped_user_id = n.deduped_user_id
  AND p.milestone = n.milestone
