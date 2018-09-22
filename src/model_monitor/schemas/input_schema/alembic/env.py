from __future__ import with_statement

import traceback
from alembic import context
from sqlalchemy import pool
from logging.config import fileConfig
from sqlalchemy import create_engine

from model_monitor.schemas.input_schema import Base
from model_monitor.io.shared import read_config, get_sqlalchemy_engine_by_conn_id

# construct fileconfig context
config = context.config
fileConfig(config.config_file_name)
target_metadata = Base.metadata

# get conn_id from mm_config.yaml
try:
    conn_id = read_config()['runtime_settings']['mm_input_conn_id']
    url = get_sqlalchemy_engine_by_conn_id(conn_id=conn_id).url
except Exception:
    print("Failed to construct SqlAlchemy URL from Airflow conn_id; \n"
          "Confirm that mm_input_conn_id is specified in config, and conn_id is configured in Airflow database")

    print(traceback.format_exc())
    exit(0)


def run_migrations_offline():
    """
    Perform offline migrations for output model_monitor schema
    """
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        version_table='mm_input_schema_versions',
        version_table_schema='public'
    )

    # run migrations
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """
    Perform online migrations for output model_monitor schema
    """
    connectable = create_engine(
        url,
        poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table='mm_input_schema_versions',
            version_table_schema='public',
            include_schemas=True
        )

        # run migrations
        with context.begin_transaction():
            context.run_migrations()


# specify online or offline mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
