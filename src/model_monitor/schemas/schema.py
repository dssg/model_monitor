from sqlalchemy import (
    Column,
    Boolean,
    Integer,
    Text,
    DateTime,
    Interval,
    Float,
    DDL,
    event,
    ForeignKey,
    Index
)

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

event.listen(
    Base.metadata,
    'before_create',
    DDL("CREATE SCHEMA IF NOT EXISTS model_monitor;")
)


class DistributionMetadata(Base):
    """
    Distribution metadata table definition

    Each row defines behavior for characterizing a distribution's CDF, including preprocessing settings,
    distribution fitting hyperparameters, and flagged warnings.
    """
    __tablename__ = 'distribution_metadata'
    __table_args__ = {'schema': 'model_monitor'}

    distribution_metadata_id = Column(Integer, primary_key=True, autoincrement=True)
    default_type = Column(Text, nullable=True)
    default_value = Column(Text, nullable=True)
    use_default_value_on_unsafe_cast = Column(Boolean, nullable=True)
    is_online = Column(Boolean, nullable=True)
    warn_on_online_support_change = Column(Boolean, nullable=True)
    is_discrete = Column(Boolean)
    support_maximum = Column(Float, nullable=True)
    support_minimum = Column(Float, nullable=True)
    remove_samples_out_of_support = Column(Boolean, nullable=True)
    is_nullable = Column(Boolean)
    tracking_mode = Column(Text)
    n_quantiles = Column(Integer, nullable=True)
    n_lower_tail_quantiles = Column(Integer, nullable=True)
    n_upper_tail_quantiles = Column(Integer, nullable=True)
    custom_quantiles = Column(Text, nullable=True)
    histogram_min = Column(Float, nullable=True)
    histogram_max = Column(Float, nullable=True)
    left_inclusive = Column(Boolean, nullable=True)
    count_outside_range = Column(Boolean, nullable=True)
    n_histogram_bins = Column(Integer, nullable=True)
    custom_histogram_bins = Column(Text, nullable=True)
    n_clusters = Column(Integer, nullable=True)
    clustering_algorithm = Column(Text, nullable=True)
    clustering_algorithm_kwargs = Column(JSONB, nullable=True)
    parametric_family = Column(Text, nullable=True)

    idx_distribution_metadata = Index('distribution_metadata_id', unique=True)


class RandomVariables(Base):
    """
    Random variable table definition

    Each row defines a random variable and the metadata necessary to process it, including the distribution metadata,
    the source table, and information about latent variable relationships.
    """
    __tablename__ = 'random_variables'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer, primary_key=True, autoincrement=True)
    rv_name = Column(Text)
    rv_type = Column(Text)
    distribution_metadata_id = Column(Integer,
                                      ForeignKey("distribution_metadata.distribution_metadata_id"),
                                      nullable=True)
    source_table = Column(Text, nullable=True)
    latent_variable_name = Column(Integer, nullable=True)
    agg_func = Column(Text, nullable=True)
    time_agg = Column(Text, nullable=True)

    idx_random_variables = Index('rv_id', unique=True)


class Moments(Base):
    """
    Moments table definition

    Each row defines, for a given RV, date, and model, a moment calculation.
    """
    __tablename__ = 'moments'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    moment_index = Column(Integer, primary_key=True)
    moment_value = Column(Float, nullable=True)

    idx_moments = Index('rv_id', 'model_id', 'as_of_date', 'moment_index', unique=True)


class Quantiles(Base):
    """
    Quantiles table definition

    Each row defines, for a given RV, date, and model, a moment calculation.
    """
    __tablename__ = 'quantiles'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    quantile = Column(Float, primary_key=True)
    quantile_value = Column(Float, nullable=True)

    idx_quantiles = Index('rv_id', 'model_id', 'as_of_date', 'quantile', unique=True)


class Histograms(Base):
    """
    Histograms table definition

    Each row defines, for a given RV, date, and model, a fixed histogram window frequency.
    """
    __tablename__ = 'histograms'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    bin_min = Column(Float, primary_key=True)
    bin_max = Column(Float, primary_key=True)
    value_count = Column(Integer, nullable=True)
    value_frequency = Column(Float, nullable=True)

    idx_histograms = Index('rv_id', 'model_id', 'as_of_date', 'bin_min', 'bin_max', unique=True)


class ClusterKdes(Base):
    """
    Cluster KDE table definition

    Each row defines, for a given RV, date, and model, a characterization of a single cluster in the distribution.
    """
    __tablename__ = 'cluster_kdes'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    cluster_center = Column(Float, primary_key=True)
    cluster_parameter = Column(Float)

    idx_cluster_kdes = Index('rv_id', 'model_id', 'as_of_date', 'cluster_center', unique=True)


class ParametricEstimates(Base):
    """
    Parametric estimates table definition

    Each row defines, for a given RV, date, and model, a numerical estimate of given parameters from a known
    statistical distribution
    """

    __tablename__ = 'parametric_estimates'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    as_of_date = Column(DateTime, primary_key=True)
    parameter_index = Column(Integer, primary_key=True)
    parameter_value = Column(Float, primary_key=True)

    idx_parametric_estimates = Index('rv_id', 'model_id', 'as_of_date', 'parameter_index', unique=True)


class MetricDefs(Base):
    """
    Metric definitions table definition

    Each row defines a single metric calculation, including the name, comparison interval, and subset arguments.
    """
    __tablename__ = 'metric_defs'
    __table_args__ = {'schema': 'model_monitor'}

    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    metric_calc_name = Column(Text)
    compare_interval = Column(Interval)
    subset_name = Column(Text)
    subset_args = Column(JSONB, nullable=True)

    idx_metric_defs = Index('metric_id', unique=True)


class Metrics(Base):
    """
    Metrics table

    Each row defines the result of a metric calculation given a date, model, random variable, and metric definition.
    """
    __tablename__ = 'metrics'
    __table_args__ = {'schema': 'model_monitor'}

    as_of_date = Column(DateTime, primary_key=True)
    model_id = Column(Integer, primary_key=True)
    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    metric_id = Column(Integer,
                       ForeignKey("metric_defs.metric_id"),
                       primary_key=True)
    metric_value = Column(Float)

    idx_metrics = Index('as_of_date', 'model_id', 'rv_id', 'metric_id', unique=True)
