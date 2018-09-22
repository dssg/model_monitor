import os.path

import alembic.config

from .schema import (
    Base,
    DistributionMetadata,
    RandomVariables,
    Moments,
    Quantiles,
    Histograms,
    ClusterKdes,
    MetricDefs,
    Metrics
)

__all__ = (
    'Base',
    'DistributionMetadata',
    'RandomVariables',
    'Moments',
    'Quantiles',
    'Histograms',
    'ClusterKdes',
    'MetricDefs',
    'Metrics',
    'mark_db_as_upgraded',
    'upgrade_db',
)


def _base_alembic_args():
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    alembic_ini_path = os.path.join(dir_path, 'alembic.ini')
    print(os.path.exists(alembic_ini_path))
    base = ['-c', alembic_ini_path]
    return base


def upgrade_db():
    args = _base_alembic_args() + ['--raiseerr', 'upgrade', 'head']
    alembic.config.main(argv=args)


def mark_db_as_upgraded():
    args = _base_alembic_args() + ['--raiseerr', 'stamp', 'head']
    alembic.config.main(argv=args)
