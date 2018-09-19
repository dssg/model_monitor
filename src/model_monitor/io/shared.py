import os
import json
import warnings
import inspect
from collections import namedtuple

import yaml
import boto3
import sqlalchemy as sa
from sqlalchemy.pool import NullPool
from sqlalchemy.engine import Engine

from airflow.hooks.base_hook import BaseHook
from airflow.models import Connection


def _n_dirname(fname, n):
    """
    Retrieve the nth parent directory of a given file

    :param fname: str, filename
    :param n: int, number of parent directories
    :return: directory name
    """
    for _ in range(n):
        fname = os.path.dirname(fname)
    return fname


def read_config(fname=None):
    """
    Read mm_config.yaml, from default location if unspecified

    :param fname: str, full path filename, defaults to mm_config.yaml in repo root directory
    :return: dict
    """
    if not fname:
        fname = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'mm_config.yaml'
        )
    with open(fname, mode='r') as f:
        return yaml.load(f)


def read_json_schema(fname='input_schema.json'):
    """
    Load json schema file from model_monitor/schemas directory

    :param fname: str, local filename in model_monitor/schemas directory
    :return: dict
    """
    if not os.path.exists(fname):
        fname = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'schemas', fname)

    with open(fname, mode='r') as f:
        return json.load(f)


def get_default_args(obj_name):
    """
    Parse full object name and extract default arguments

    :param obj_name: str, object name (ex: import {object_name})
    :return: dict
    """
    try:
        # dynamically parse and import the target object
        module_name = '.'.join(obj_name.split('.')[:-1])
        class_name = obj_name.split('.')[-1]
        mod = __import__(module_name, fromlist=[class_name])
        obj = getattr(mod, class_name)

        # inspect and return components
        signature = inspect.signature(obj)
        return {k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty}
    except ImportError:
        warnings.warn("Cannot find default arguments, skipping `{}`...".format(obj_name))
        return dict()


def get_sqlalchemy_engine_by_conn_id(conn_id):
    """
    Create SqlAlchemy engine from Airflow connection ID credentials

    :param conn_id: str, connection ID to identify Airflow connection
    :return: SqlAlchemy engine
    """

    conn = BaseHook.get_connection(conn_id)  # type: Connection
    assert conn.conn_type == 'postgres', "conn_type must be 'postgres', got '' instead".format(conn.conn_type)

    # if schema search path specified in extra, apply to engine
    if conn.extra_dejson:
        connect_args = {'options': '-csearch_path={}'.format(conn.extra_dejson["schema_search_path"])}
    # else, use default schema search path
    else:
        connect_args = {}

    return sa.create_engine(
        "postgresql://{user}:{pwd}@{host}:{port}/{db}".format(  # type: Engine
            user=conn.login,
            pwd=conn.password,
            host=conn.host,
            port=conn.port,
            db=conn.schema
        ),
        poolclass=NullPool,
        connect_args=connect_args
    )


def get_s3_client_by_conn_id(conn_id):
    """
    Create s3 boto client from Airflow connection ID credentials

    :param conn_id: str, connection ID to identify Airflow connection
    :return: boto3 client
    """

    conn = BaseHook.get_connection(conn_id)  # type: Connection
    assert conn.conn_type == 's3'
    conn_extra = conn.extra_dejson

    return boto3.client(
        's3',
        aws_access_key_id=conn_extra['aws_access_key_id'],
        aws_secret_access_key=conn_extra['aws_secret_access_key']
    )


DistributionMetadata = namedtuple('DistributionMetadata', [
    'distribution_metadata_id',
    'default_type',
    'default_value',
    'use_default_value_on_unsafe_cast',
    'is_online',
    'warn_on_online_support_change',
    'is_discrete',
    'support_maximum',
    'support_minimum',
    'remove_samples_out_of_support',
    'is_nullable',
    'tracking_mode',
    'n_quantiles',
    'n_lower_tail_quantiles',
    'n_upper_tail_quantiles',
    'custom_quantiles',
    'histogram_min',
    'histogram_max',
    'left_inclusive',
    'count_outside_range',
    'n_histogram_bins',
    'custom_histogram_bins',
    'n_clusters',
    'clustering_algorithm',
    'clustering_algorithm_kwargs',
    'parametric_family',
])
