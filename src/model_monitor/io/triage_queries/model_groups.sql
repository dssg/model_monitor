/*
Default query for model_groups using triage results-schema

Assumptions:
- Schema search order specified in connection
 */

SELECT
    model_group_id,
    model_type,
    model_parameters,
    feature_list,
    model_config
FROM model_groups mg
WHERE (mg.model_group_id IN(SELECT (UNNEST(%(included_model_group_ids)s))) OR
       NOT %(filter_included_model_group_ids)s)
AND (mg.model_group_id NOT IN (SELECT (UNNEST(%(excluded_model_group_ids)s))) OR
       NOT %(filter_excluded_model_group_ids)s)
AND (mg.model_type IN (SELECT (UNNEST(%(included_model_types)s))) OR
       NOT %(filter_included_model_types)s)
AND (mg.model_type NOT IN (SELECT (UNNEST(%(excluded_model_types)s))) OR
       NOT %(filter_excluded_model_types)s);
