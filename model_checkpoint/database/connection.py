"""Database connection and operations - optimized legacy compatibility"""

from .base_connection import BaseDatabaseConnection


class DatabaseConnection(BaseDatabaseConnection):
    """Simple SQLite database connection - inherits all optimized functionality from base class

    This class now provides 100% backward compatibility while using the optimized
    base implementation. All CRUD operations are inherited from BaseDatabaseConnection.
    """

    def __init__(self, database_url: str = "sqlite:///experiments.db"):
        """Initialize database connection with legacy compatibility"""
        # Initialize base class (includes all optimized functionality)
        super().__init__(database_url)

    # All methods (save_experiment, get_experiment, save_metric, get_metrics,
    # save_checkpoint, get_checkpoint) are inherited from BaseDatabaseConnection
    # This eliminates 150+ lines of duplicate code while maintaining compatibility