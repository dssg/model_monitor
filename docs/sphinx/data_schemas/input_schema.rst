.. _input-schema:

Input Schema
==================

The following schema specifies legal input data into ``model_monitor``. The following key constraints must be enforced:

- Each ``model_group_id`` is associated with a single model class and set of hyperparameters
- Each ``model_id`` is associated with a single ``model_group_id`` and ``train_matrix_id``

NOTE: the foreign key relationship between ``train_matrix_id`` and ``test_matrix_id`` is nullabe. Although each training
matrix can have a default ``test_matrix_id`` associated with it, the relationship can be one-to-many in the
``predictions`` table. In other words, you can test multiple different testing matrices on the same training matrix.


``model_groups``:

.. table::

    +----------------+-----------+----------------------------------------------------+--------+-----------+
    |      name      |   dtype   |                    description                     |nullable|constraints|
    +================+===========+====================================================+========+===========+
    |model_group_id  |Integer    |model group ID, unique identifier                   |False   |PK         |
    +----------------+-----------+----------------------------------------------------+--------+-----------+
    |model_type      |Text       |model class definition, must be importable in python|False   |           |
    +----------------+-----------+----------------------------------------------------+--------+-----------+
    |model_parameters|JSONB      |model parameters passed to model_type, by keyword   |True    |           |
    +----------------+-----------+----------------------------------------------------+--------+-----------+
    |feature_list    |ARRAY(Text)|text list of feature names, comma delimited         |False   |           |
    +----------------+-----------+----------------------------------------------------+--------+-----------+

``models``:

.. table::

    +---------------+-------+---------------------------+--------+-----------+
    |     name      | dtype |        description        |nullable|constraints|
    +===============+=======+===========================+========+===========+
    |model_id       |Integer|model ID, unique identifier|False   |PK         |
    +---------------+-------+---------------------------+--------+-----------+
    |model_group_id |Integer|model group ID             |False   |FK         |
    +---------------+-------+---------------------------+--------+-----------+
    |train_matrix_id|Integer|training matrix unique ID  |False   |FK         |
    +---------------+-------+---------------------------+--------+-----------+

``training_matrices``:

.. table::

    +-----------------+--------+----------------------------------+--------+-----------+
    |      name       | dtype  |           description            |nullable|constraints|
    +=================+========+==================================+========+===========+
    |train_matrix_id  |Integer |training matrix unique ID         |False   |PK         |
    +-----------------+--------+----------------------------------+--------+-----------+
    |train_matrix_uuid|Text    |training matrix UUID              |False   |           |
    +-----------------+--------+----------------------------------+--------+-----------+
    |train_as_of_date |DateTime|training matrix most recent update|False   |           |
    +-----------------+--------+----------------------------------+--------+-----------+

``testing_matrices``:

.. table::

    +----------------+--------+------------------------------------+--------+-----------+
    |      name      | dtype  |            description             |nullable|constraints|
    +================+========+====================================+========+===========+
    |test_matrix_id  |Integer |testing matrix unique ID            |False   |PK         |
    +----------------+--------+------------------------------------+--------+-----------+
    |test_matrix_uuid|Text    |testing matrix UUID                 |False   |           |
    +----------------+--------+------------------------------------+--------+-----------+
    |test_as_of_date |DateTime|testing matrix most recent update   |False   |           |
    +----------------+--------+------------------------------------+--------+-----------+
    |train_matrix_id |Integer |associated training matrix unique ID|True    |FK         |
    +----------------+--------+------------------------------------+--------+-----------+

``predictions``:

.. table::

    +--------------+-------+------------------------+--------+-----------+
    |     name     | dtype |      description       |nullable|constraints|
    +==============+=======+========================+========+===========+
    |model_id      |Integer|model ID                |False   |PK, FK     |
    +--------------+-------+------------------------+--------+-----------+
    |test_matrix_id|Text   |testing matrix unique ID|False   |PK         |
    +--------------+-------+------------------------+--------+-----------+
    |entity_id     |Integer|entity unique ID        |False   |PK         |
    +--------------+-------+------------------------+--------+-----------+
    |score         |Float  |predicted score         |False   |           |
    +--------------+-------+------------------------+--------+-----------+
    |label         |Integer|true outcome            |False   |           |
    +--------------+-------+------------------------+--------+-----------+

``feature_importances``:

.. table::

    +------------------+-------+------------------------------+--------+-----------+
    |       name       | dtype |         description          |nullable|constraints|
    +==================+=======+==============================+========+===========+
    |model_id          |Integer|model ID                      |False   |PK, FK     |
    +------------------+-------+------------------------------+--------+-----------+
    |feature           |Text   |feature name, unique ID       |False   |PK         |
    +------------------+-------+------------------------------+--------+-----------+
    |feature_importance|Float  |estimated feature contribution|False   |           |
    +------------------+-------+------------------------------+--------+-----------+

