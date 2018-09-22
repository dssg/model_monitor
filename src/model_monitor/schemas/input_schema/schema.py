from sqlalchemy import (
    Column,
    Integer,
    Text,
    DateTime,
    Float
)

from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ModelGroups(Base):
    """
    Table definition for 'model_groups'

    Each row is specified by a single model class object, set of hyperparameters, and list of features
    """
    __tablename__ = "model_groups"

    model_group_id = Column(Integer, primary_key=True)
    model_type = Column(Text)
    model_parameters = Column(JSONB)
    feature_list = Column(ARRAY(Text, dimensions=1))


class Models(Base):
    """
    Table definition for 'models'

    Each row is specified by a single model group, training date, and training matrix ID
    """
    __tablename__ = 'models'

    model_id = Column(Integer, primary_key=True)
    model_group_id = Column(Integer)
    train_matrix_id = Column(Integer)


class TrainingMatrices(Base):
    """
    Table definition for 'training_matrices'

    Each row is specified by a single matrix UUID and most recent matrix update
    """

    __tablename__ = 'training_matrices'

    train_matrix_id = Column(Integer, primary_key=True)
    train_matrix_uuid = Column(Text)
    train_as_of_date = Column(DateTime)


class TestingMatrices(Base):
    """
    Table definition for 'testing_matrices'

    Each row is specified by a single matrix UUID, most recent matrix update, and optional associated training matrix
    """
    test_matrix_id = Column(Integer, primary_key=True)
    test_matrix_uuid = Column(Text)
    test_as_of_date = Column(DateTime)
    train_matrix_id = Column(Integer, nullable=True)


class Predictions(Base):
    """
    Table definition for 'predictions'

    Each row is specified by a single model, testing matrix ID, and entity ID
    """
    __tablename__ = 'predictions'

    model_id = Column(Integer, primary_key=True)
    test_matrix_id = Column(Text, primary_key=True)
    entity_id = Column(Integer, primary_key=True)
    score = Column(Float)
    label = Column(Integer)


class FeatureImportances(Base):
    """
    Table definition for 'feature_importances'

    Each row is specified by a single model and feature name
    """
    __tablename__ = 'feature_importances'

    model_id = Column(Integer, primary_key=True)
    feature = Column(Text, primary_key=True)
    feature_importance = Column(Float)
