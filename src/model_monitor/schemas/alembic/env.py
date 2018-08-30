from __future__ import with_statement

from alembic import context
from sqlalchemy import pool
from logging.config import fileConfig
from sqlalchemy import create_engine

from model_monitor.schemas import Base
from model_monitor.io.shared import read_config, get_sqlalchemy_engine_by_conn_id

# construct fileconfig context
config = context.config
fileConfig(config.config_file_name)
target_metadata = Base.metadata

# get conn_id from alembic.ini
conn_id = read_config()['runtime_settings']['mm_output_conn_id']
url = get_sqlalchemy_engine_by_conn_id(conn_id=conn_id).url


def include_object(object, name, type_, reflected, compare_to):
    """
    Checks whether to version control a sqlalchemy object

    NOTE: signature is defined by alembic, so cannot change first argument that shadows outer scope
    """
    # Note: this schema is necessary for alembic, even though it shadows built-ins
    return object.schema == 'model_monitor'


def run_migrations_offline():
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        version_table='mm_results_schema_versions',
        version_table_schema='public',
        include_object=include_object
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = create_engine(
        url,
        poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table='mm_results_schema_versions',
            version_table_schema='public',
            include_schemas=True,
            include_object=include_object
        )
        connection.execute('set search_path to "{}", public'.format('model_monitor'))

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
