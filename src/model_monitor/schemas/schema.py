from sqlalchemy import (
    Column,
    Boolean,
    Integer,
    Text,
    DateTime,
    Interval,
    Float,
    DDL,
    event
)

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base

DEFAULT_QUERY_LOC = "../../import_tool/triage_default_queries/"
Base = declarative_base()

event.listen(
    Base.metadata,
    'before_create',
    DDL("CREATE SCHEMA IF NOT EXISTS model_monitor;")
)


class DistributionMetadata(Base):
    __tablename__ = 'distribution_metadata'
    __table_args__ = {'schema': 'model_monitor'}

    distribution_id = Column(Integer, primary_key=True)
    default_type = Column(Text, nullable=True)
    default_value = Column(Text, nullable=True)
    use_default_value_on_unsafe_cast = Column(Boolean, nullable=True)
    is_online = Column(Boolean, nullable=True)
    warn_on_online_support_change = Column(Boolean, nullable=True)
    is_discrete = Column(Boolean)
    custom_discrete_support = Column(Text, nullable=True)
    support_maximum = Column(Float, nullable=True)
    support_minimum = Column(Float, nullable=True)
    remove_support_violators = Column(Boolean, nullable=True)
    is_nullable = Column(Boolean)
    track_moments = Column(Boolean)
    n_moments = Column(Integer, nullable=True)
    n_central_moments = Column(Integer, nullable=True)
    n_standardized_moments = Column(Integer, nullable=True)
    track_quantiles = Column(Boolean)
    n_quantiles = Column(Integer, nullable=True)
    n_lower_tail_quantiles = Column(Integer, nullable=True)
    n_upper_tail_quantiles = Column(Integer, nullable=True)
    custom_quantiles = Column(Text, nullable=True)
    track_histogram = Column(Boolean)
    histogram_min = Column(Float, nullable=True)
    histogram_max = Column(Float, nullable=True)
    left_inclusive = Column(Boolean, nullable=True)
    count_outside_range = Column(Boolean, nullable=True)
    n_histogram_bins = Column(Integer, nullable=True)
    custom_histogram_bins = Column(Text, nullable=True)
    track_cluster_estimate = Column(Boolean)
    num_clusters = Column(Integer, nullable=True)
    bandwidth = Column(Float, nullable=True)
    kernel_type = Column(Text, nullable=True)
    track_parametric_estimate = Column(Boolean, nullable=True)
    parametric_family = Column(Text, nullable=True)


class RandomVariables(Base):
    __tablename__ = 'random_variables'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer, primary_key=True)
    rv_name = Column(Text)
    rv_type = Column(Text)
    distribution_metadata_id = Column(Integer, nullable=True)
    is_reserved = Column(Boolean, nullable=True)
    source_table = Column(Text)
    latent_variable_name = Column(Integer)
    agg_func = Column(Text)
    time_agg = Column(Text)


class Moments(Base):
    __tablename__ = 'moments'
    __table_args__ = {'schema': 'model_monitor'}

    feature_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    moment_index = Column(Integer, primary_key=True)
    moment_value = Column(Float, nullable=True)


class Quantiles(Base):
    __tablename__ = 'quantiles'
    __table_args__ = {'schema': 'model_monitor'}

    feature_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    quantile = Column(Float, primary_key=True)
    quantile_value = Column(Float, nullable=True)


class Histograms(Base):
    __tablename__ = 'histograms'
    __table_args__ = {'schema': 'model_monitor'}

    feature_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    bin_min = Column(Float, primary_key=True)
    bin_max = Column(Float, primary_key=True)
    value_count = Column(Integer, nullable=True)
    value_frequency = Column(Float, nullable=True)


class ClusterKdes(Base):
    __tablename__ = 'cluster_kdes'
    __table_args__ = {'schema': 'model_monitor'}

    feature_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    cluster_center = Column(Float, primary_key=True)
    cluster_bandwidth = Column(Float)


class MetricDefs(Base):
    __tablename__ = 'metric_defs'
    __table_args__ = {'schema': 'model_monitor'}

    metric_id = Column(Integer, primary_key=True)
    metric_calc_name = Column(Text)
    compare_interval = Column(Interval)
    subset_name = Column(Text)
    threshold = Column(Float, nullable=True)
    weights_args = Column(JSONB, nullable=True)


class Metrics(Base):
    __tablename__ = 'metrics'
    __table_args__ = {'schema': 'model_monitor'}

    as_of_date = Column(DateTime, primary_key=True)
    model_id = Column(Integer, primary_key=True)
    rv_id = Column(Integer, primary_key=True)
    metric_id = Column(Integer, primary_key=True)
    metric_value = Column(Float)
