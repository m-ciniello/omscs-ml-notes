WITH active_users AS (
  SELECT DISTINCT deduped_user_id
  FROM `sdp-prd-shop-ml.mart.mart__shop_app__deduped_user_dimension`
  CROSS JOIN UNNEST(activity) AS activity
  WHERE activity.date BETWEEN DATE '{{ reference_date }}' - INTERVAL 30 DAY
    AND DATE '{{ reference_date }}' - INTERVAL 1 DAY
),

eligible_users AS (
  SELECT up.deduped_user_id
  FROM `sdp-prd-shop-ml.mart.mart__shop_app__user_profile` up
  JOIN active_users USING(deduped_user_id)
  WHERE (up.last_shop_app_user_geo.country_code IN ('US') OR up.country_code IN ('US'))
    AND up.risk_level IN ('safe/not rated', 'safe', 'low_risk')
    AND NOT up.is_on_giveaways_blocklist
    AND NOT up.is_shopifolk_user
),

prior_order_counts AS (
  SELECT
    eu.deduped_user_id,
    COUNT(DISTINCT o.order_id) AS orders_before
  FROM eligible_users eu
  LEFT JOIN `sdp-prd-shop-ml.mart.mart__shared_data__shop_attributed_order_facts` o
    ON eu.deduped_user_id = o.deduped_user_id
    AND o.created_at < TIMESTAMP '{{ reference_date }}'
    AND o.attribution_type IN ('native', 'referral')
  GROUP BY 1
),

user_segments AS (
  SELECT
    deduped_user_id,
    orders_before,
    CASE
      WHEN orders_before = 0 THEN '0 orders (new buyers)'
      WHEN orders_before = 1 THEN '1 order (1→2 conversion)'
      ELSE '2+ orders (repeat buyers)'
    END AS segment
  FROM prior_order_counts
),

post_conversions AS (
  SELECT
    us.deduped_user_id,
    us.segment,
    LOGICAL_OR(
      nof.created_at BETWEEN TIMESTAMP '{{ reference_date }}'
        AND TIMESTAMP_ADD(TIMESTAMP '{{ reference_date }}', INTERVAL 7 DAY)
    ) AS converted_7d,
    LOGICAL_OR(
      nof.created_at BETWEEN TIMESTAMP '{{ reference_date }}'
        AND TIMESTAMP_ADD(TIMESTAMP '{{ reference_date }}', INTERVAL 10 DAY)
    ) AS converted_10d,
    LOGICAL_OR(
      nof.created_at BETWEEN TIMESTAMP '{{ reference_date }}'
        AND TIMESTAMP_ADD(TIMESTAMP '{{ reference_date }}', INTERVAL 14 DAY)
    ) AS converted_14d
  FROM user_segments us
  LEFT JOIN `sdp-prd-shop-ml.mart.mart__shared_data__shop_attributed_order_facts` nof
    ON us.deduped_user_id = nof.deduped_user_id
    AND nof.created_at BETWEEN TIMESTAMP '{{ reference_date }}'
      AND TIMESTAMP_ADD(TIMESTAMP '{{ reference_date }}', INTERVAL 14 DAY)
    AND nof.attribution_type IN ('native', 'referral')
  GROUP BY 1, 2
)

SELECT
  segment,
  COUNT(1) AS total_users,
  COUNTIF(converted_7d) AS conversions_7d,
  COUNTIF(converted_10d) AS conversions_10d,
  COUNTIF(converted_14d) AS conversions_14d,
  SAFE_DIVIDE(COUNTIF(converted_7d), COUNT(1)) AS conversion_rate_7d,
  SAFE_DIVIDE(COUNTIF(converted_10d), COUNT(1)) AS conversion_rate_10d,
  SAFE_DIVIDE(COUNTIF(converted_14d), COUNT(1)) AS conversion_rate_14d
FROM post_conversions
GROUP BY 1

UNION ALL

SELECT
  'All users' AS segment,
  COUNT(1) AS total_users,
  COUNTIF(converted_7d) AS conversions_7d,
  COUNTIF(converted_10d) AS conversions_10d,
  COUNTIF(converted_14d) AS conversions_14d,
  SAFE_DIVIDE(COUNTIF(converted_7d), COUNT(1)) AS conversion_rate_7d,
  SAFE_DIVIDE(COUNTIF(converted_10d), COUNT(1)) AS conversion_rate_10d,
  SAFE_DIVIDE(COUNTIF(converted_14d), COUNT(1)) AS conversion_rate_14d
FROM post_conversions

ORDER BY 1
