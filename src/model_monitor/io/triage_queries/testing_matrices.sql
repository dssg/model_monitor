SELECT
    tem.test_matrix_id,
    tem.test_matrix_uuid,
    tem.test_as_of_date,
    tem.train_matrix_id
FROM testing_matrices tem
INNER JOIN training_matrices trm
ON tem.train_matrix_id = trm.test_matrix_id
INNER JOIN models m
ON m.train_matrix_id = trm.train_matrix_id
INNER JOIN model_groups mg
ON mg.model_group_id = m.model_group_id
WHERE (mg.model_group_id IN(SELECT (UNNEST(%(included_model_group_ids)s))) OR
       NOT %(filter_included_model_group_ids)s)
AND (mg.model_group_id NOT IN (SELECT (UNNEST(%(excluded_model_group_ids)s))) OR
       NOT %(filter_excluded_model_group_ids)s)
AND (mg.model_type IN (SELECT (UNNEST(%(included_model_types)s))) OR
       NOT %(filter_included_model_types)s)
AND (mg.model_type NOT IN (SELECT (UNNEST(%(excluded_model_types)s))) OR
       NOT %(filter_excluded_model_types)s)
AND (m.model_id IN(SELECT (UNNEST(%(included_model_ids)s))) OR
       NOT %(filter_included_model_ids)s)
AND (m.model_id NOT IN (SELECT (UNNEST(%(excluded_model_ids)s))) OR
       NOT %(filter_excluded_model_ids)s)
AND trm.train_as_of_date >= %(train_as_of_date_start)
AND trm.train_as_of_date <= %(train_as_of_date_end)
AND tem.test_as_of_date >= %(test_as_of_date_start)
AND tem.test_as_of_date <= %(test_as_of_date_end);