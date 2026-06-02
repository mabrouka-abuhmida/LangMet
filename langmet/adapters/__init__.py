"""Adapters for integrating LangMet with storage and frameworks."""

from .memory import InMemoryMetricsRepository

__all__ = ["InMemoryMetricsRepository"]

# SQLAlchemy is an optional dependency; only expose the adapter when available.
try:
    from .sqlalchemy_repo import SQLAlchemyMetricsRepository  # noqa: F401

    __all__.append("SQLAlchemyMetricsRepository")
except ImportError:  # pragma: no cover - depends on optional extra
    pass
