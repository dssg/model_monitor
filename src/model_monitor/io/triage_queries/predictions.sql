/*
Default query for predictions using triage results-schema

Assumptions:
- Schema search order specified in connection
- Each model_id has a unique associated model_group_id
 */

SELECT
    m.model_id,
    m.model_group_id,
    as_of_date,
    score,
    label_value,
    rank_abs,
    rank_pct
FROM predictions p
INNER JOIN models m
ON m.model_id = p.model_id
INNER JOIN model_groups mg
ON mg.model_group_id = m.model_group_id
WHERE as_of_date = %(target_date)s
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

