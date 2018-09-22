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

from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

event.listen(
    Base.metadata,
    'before_create',
    DDL("CREATE SCHEMA IF NOT EXISTS model_monitor;")
)


class DistributionMetadata(Base):
    """
    Table definition for 'distribution_metadata'

    Each row defines behavior for characterizing a distribution's CDF, including preprocessing settings,
    distribution fitting hyperparameters, and flagged warnings.
    """
    __tablename__ = 'distribution_metadata'
    __table_args__ = {'schema': 'model_monitor'}

    distribution_metadata_id = Column(Integer, primary_key=True, autoincrement=True)
    is_discrete = Column(Boolean)
    is_nullable = Column(Boolean)
    default_type = Column(Text, nullable=True)
    default_value = Column(Float, nullable=True)
    use_default_value_on_unsafe_cast = Column(Boolean, nullable=True)
    support_minimum = Column(Float, nullable=True)
    support_maximum = Column(Float, nullable=True)
    remove_samples_out_of_support = Column(Boolean, nullable=True)
    is_online = Column(Boolean, nullable=True)
    warn_on_online_support_change = Column(Boolean, nullable=True)
    interpolation_mode = Column(Text, nullable=True)
    tracking_mode = Column(Text, nullable=True)
    n_quantiles = Column(Integer, nullable=True)
    n_lower_tail_quantiles = Column(Integer, nullable=True)
    n_upper_tail_quantiles = Column(Integer, nullable=True)
    custom_quantiles = Column(Text, nullable=True)
    histogram_min = Column(Float, nullable=True)
    histogram_max = Column(Float, nullable=True)
    n_histogram_bins = Column(Integer, nullable=True)
    custom_histogram_bins = Column(ARRAY(Float), nullable=True)
    n_clusters = Column(Integer, nullable=True)
    clustering_algorithm = Column(Text, nullable=True)
    clustering_algorithm_kwargs = Column(JSONB, nullable=True)
    clustering_parametric_family = Column(Text, nullable=True)
    parametric_family = Column(Text, nullable=True)

    idx_distribution_metadata = Index('distribution_metadata_id', unique=True)


class RandomVariables(Base):
    """
    Table definition for 'random_variables'

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
    is_test = Column(Boolean, nullable=True)
    latent_variable_name = Column(Integer, nullable=True)
    agg_func = Column(Text, nullable=True)
    time_agg = Column(Text, nullable=True)

    idx_random_variables = Index('rv_id', unique=True)


class Quantiles(Base):
    """
    Table definition for 'quantiles'

    Each row defines, for a given RV and model, a quantile calculation.
    """
    __tablename__ = 'quantiles'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    quantile = Column(Float, primary_key=True)
    quantile_value = Column(Float, nullable=True)

    idx_quantiles = Index('rv_id', 'model_id', 'quantile', unique=True)


class Histograms(Base):
    """
    Table definition for 'histograms'

    Each row defines, for a given RV and model, a fixed histogram window frequency.
    """
    __tablename__ = 'histograms'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    bin_min = Column(Float, primary_key=True)
    bin_max = Column(Float, primary_key=True)
    value_count = Column(Integer, nullable=True)
    value_frequency = Column(Float, nullable=True)

    idx_histograms = Index('rv_id', 'model_id', 'bin_min', 'bin_max', unique=True)


class ClusterNonparametricWeights(Base):
    """
    Table definition for 'cluster_nonparametric_weights'

    Each row defines, for a given RV and model, the cluster center and cluster weight
    """
    __tablename__ = 'cluster_nonparametric_weights'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    cluster_center = Column(Float, primary_key=True)
    cluster_weight = Column(Float)

    idx_cluster_nonparametric_weights = Index('rv_id', 'model_id', 'cluster_center', unique=True)


class ClusterParameters(Base):
    """
    Table definition for 'cluster_parameters'

    Each row defines, for a given RV and model, a characterization of a single cluster parameter
    """
    __tablename__ = 'cluster_parameters'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    cluster_center = Column(Float, primary_key=True)
    cluster_parameter_index = Column(Integer, primary_key=True)
    cluster_parameter_name = Column(Text)
    cluster_parameter_value = Column(Float)
    cluster_weight = Column(Float)

    idx_cluster_parameters = Index('rv_id', 'model_id', 'cluster_center', 'cluster_parameter_index',
                                   unique=True)


class ParametricEstimates(Base):
    """
    Table definition for 'parametric_estimates'

    Each row defines, for a given RV and model, a numerical estimate of given parameters from a known
    statistical distribution
    """

    __tablename__ = 'parametric_estimates'
    __table_args__ = {'schema': 'model_monitor'}

    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    model_id = Column(Integer, primary_key=True)
    parameter_index = Column(Integer, primary_key=True)
    parameter_name = Column(Text)
    parameter_value = Column(Float, primary_key=True)

    idx_parametric_estimates = Index('rv_id', 'model_id', 'parameter_index', unique=True)


class MetricDefs(Base):
    """
    Table definition for 'metric_defs'

    Each row defines a single metric calculation, including the name, comparison interval, and subset arguments.
    """
    __tablename__ = 'metric_defs'
    __table_args__ = {'schema': 'model_monitor'}

    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    metric_calc_name = Column(Text)
    compare_interval = Column(Interval)
    subset_name = Column(Text)
    subset_threshold = Column(Float, nullable=True)

    idx_metric_defs = Index('metric_id', unique=True)


class Metrics(Base):
    """
    Table definition for 'metrics'

    Each row defines the result of a metric calculation given a model, random variable, and metric definition.
    """
    __tablename__ = 'metrics'
    __table_args__ = {'schema': 'model_monitor'}

    model_id = Column(Integer, primary_key=True)
    rv_id = Column(Integer,
                   ForeignKey("random_variables.rv_id"),
                   primary_key=True)
    metric_id = Column(Integer,
                       ForeignKey("metric_defs.metric_id"),
                       primary_key=True)
    metric_value = Column(Float)

    idx_metrics = Index('model_id', 'rv_id', 'metric_id', unique=True)
