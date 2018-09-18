/*
Default query for models using triage results-schema

Assumptions:
- Schema search order specified in connection
- Each model_id has a unique associated model_group_id
 */

SELECT
    model_id,
    m.model_group_id,
    m.model_parameters,
    model_comment,
    batch_comment,
    config,
    train_end_time,
    test,
    train_label_window
FROM models m
INNER JOIN model_groups mg
ON mg.model_group_id = m.model_group_id
WHERE (m.model_group_id IN(SELECT (UNNEST(%(included_model_group_ids)s))) OR
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
