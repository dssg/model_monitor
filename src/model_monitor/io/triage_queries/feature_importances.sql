/*
Default query for feature_importances using triage results-schema

Assumptions:
- Schema search order specified in connection
- Each model_id has a unique associated model_group_id
- models.train_end_time corresponds to predictions.as_of_date
 */

SELECT
    f.model_id,
    m.model_group_id,
    m.train_end_time AS as_of_date,
    f.feature,
    f.feature_importance,
    f.rank_abs,
    f.rank_pct,
FROM feature_importances f
INNER JOIN model_ids m
ON m.model_id = f.model_id
INNER JOIN model_groups mg
ON mg.model_group_id = m.model_group_id
WHERE m.train_end_time::DATE == %(target_date)s
AND (m.model_group_id IN(SELECT (UNNEST(%(included_model_group_ids)s))) OR
       NOT %(filter_included_model_group_ids)s)
AND (m.model_group_id NOT IN (SELECT (UNNEST(%(excluded_model_group_ids)s))) OR
       NOT %(filter_excluded_model_group_ids)s)
AND (mg.model_type IN (SELECT (UNNEST(%(included_model_types)s))) OR
       NOT %(filter_included_model_types)s)
AND (mg.model_type NOT IN (SELECT (UNNEST(%(excluded_model_types)s))) OR
       NOT %(filter_excluded_model_types)s)
AND (m.model_id IN(SELECT (UNNEST(%(included_model_ids)s))) OR
       NOT %(filter_included_model_ids)s)
AND (m.model_id NOT IN (SELECT (UNNEST(%(excluded_model_ids)s))) OR
       NOT %(filter_excluded_model_ids)s);
