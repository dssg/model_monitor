import os.path

import alembic.config

from .schema import (
    Base,
    ModelGroups,
    Models,
    TrainingMatrices,
    TestingMatrices,
    Predictions,
    FeatureImportances
)

__all__ = (
    'Base',
    'Models',
    'ModelGroups',
    'TrainingMatrices',
    'TestingMatrices',
    'Predictions',
    'FeatureImportances',
    'mark_db_as_upgraded',
    'upgrade_db'
)


def _base_alembic_args():
    """
    Parse alembic.ini path for base arguments
    """
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    alembic_ini_path = os.path.join(dir_path, 'alembic.ini')
    base = ['-c', alembic_ini_path]
    return base


def upgrade_db():
    """
    Primary handle for upgrading database
    """
    args = _base_alembic_args() + ['--raiseerr', 'upgrade', 'head']
    alembic.config.main(argv=args)


def mark_db_as_upgraded():
    """
    Primary handle for marking existing database as upgraded
    """
    args = _base_alembic_args() + ['--raiseerr', 'stamp', 'head']
    alembic.config.main(argv=args)
