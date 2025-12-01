"""Enhanced database connection with migration support - optimized to eliminate redundancy"""

from typing import Any, Dict, List, Optional

from .base_connection import BaseDatabaseConnection
from .migration_manager import MigrationManager
from .models import Checkpoint


class EnhancedDatabaseConnection(BaseDatabaseConnection):
    """Enhanced SQLite database connection - inherits all base functionality, adds migrations"""

    def __init__(
        self, database_url: str = "sqlite:///experiments.db", auto_migrate: bool = True
    ):
        """Initialize enhanced database connection with migration support"""
        # Initialize base class (includes optimized table creation and CRUD)
        super().__init__(database_url)

        # Initialize migration manager
        self.migration_manager = MigrationManager(self.db_path)

        # Run migrations if enabled
        if auto_migrate:
            self._run_migrations()

    def _run_migrations(self) -> None:
        """Run pending database migrations"""
        try:
            result = self.migration_manager.migrate()
            if result["migrations_run"] > 0:
                self.logger.info(
                    f"Applied {result['migrations_run']} database migrations"
                )
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise

    # Enhanced-specific methods (not in base class)
    def get_checkpoints_by_experiment(
        self,
        experiment_id: str,
        checkpoint_type: Optional[str] = None,
        best_only: bool = False,
    ) -> List[Checkpoint]:
        """Get checkpoints with advanced filtering - enhanced-specific method"""
        with self._get_connection() as conn:
            query_parts = [
                """
                SELECT id, experiment_id, epoch, step, checkpoint_type, file_path,
                       file_size, checksum, model_name, loss, val_loss, notes,
                       is_best_loss, is_best_val_loss, is_best_metric,
                       metrics, metadata, created_at
                FROM checkpoints WHERE experiment_id = ?
            """
            ]
            params = [experiment_id]

            if checkpoint_type:
                query_parts.append("AND checkpoint_type = ?")
                params.append(checkpoint_type)

            if best_only:
                query_parts.append(
                    "AND (is_best_loss = 1 OR is_best_val_loss = 1 OR is_best_metric = 1)"
                )

            query_parts.append("ORDER BY created_at DESC")
            query = " ".join(query_parts)

            cursor = conn.execute(query, params)
            return [self._row_to_checkpoint(row) for row in cursor.fetchall()]

    def update_best_flags(
        self,
        experiment_id: str,
        checkpoint_id: str,
        is_best_loss: bool = False,
        is_best_val_loss: bool = False,
        is_best_metric: bool = False,
    ) -> None:
        """Update best model flags - enhanced-specific atomic operation"""
        with self._get_connection() as conn:
            # Optimized: clear and set in single transaction
            updates = []
            if is_best_loss:
                updates.append(
                    (
                        "UPDATE checkpoints SET is_best_loss = FALSE WHERE experiment_id = ? AND is_best_loss = TRUE",
                        (experiment_id,),
                    )
                )
            if is_best_val_loss:
                updates.append(
                    (
                        "UPDATE checkpoints SET is_best_val_loss = FALSE WHERE experiment_id = ? AND is_best_val_loss = TRUE",
                        (experiment_id,),
                    )
                )
            if is_best_metric:
                updates.append(
                    (
                        "UPDATE checkpoints SET is_best_metric = FALSE WHERE experiment_id = ? AND is_best_metric = TRUE",
                        (experiment_id,),
                    )
                )

            # Execute all updates in single transaction
            for query, params in updates:
                conn.execute(query, params)

            # Set new best checkpoint
            conn.execute(
                """
                UPDATE checkpoints SET is_best_loss = ?, is_best_val_loss = ?, is_best_metric = ?
                WHERE id = ?
            """,
                (is_best_loss, is_best_val_loss, is_best_metric, checkpoint_id),
            )

            conn.commit()

    def get_experiment_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment statistics - enhanced-specific analytics"""
        with self._get_connection() as conn:
            # Optimized: single query for experiment info
            exp_cursor = conn.execute(
                """
                SELECT name, status, start_time, end_time, step
                FROM experiments WHERE id = ?
            """,
                (experiment_id,),
            )
            exp_row = exp_cursor.fetchone()

            if not exp_row:
                return {}

            # Optimized: single query for checkpoint counts
            ckpt_cursor = conn.execute(
                """
                SELECT checkpoint_type, COUNT(*)
                FROM checkpoints WHERE experiment_id = ?
                GROUP BY checkpoint_type
            """,
                (experiment_id,),
            )
            checkpoint_counts = dict(ckpt_cursor.fetchall())

            # Optimized: single query for metrics summary
            metrics_cursor = conn.execute(
                """
                SELECT metric_name, COUNT(*), MIN(metric_value), MAX(metric_value), AVG(metric_value)
                FROM metrics WHERE experiment_id = ?
                GROUP BY metric_name
            """,
                (experiment_id,),
            )

            metrics_summary = {
                row[0]: {"count": row[1], "min": row[2], "max": row[3], "avg": row[4]}
                for row in metrics_cursor.fetchall()
            }

            # Calculate duration efficiently
            duration = exp_row[3] - exp_row[2] if exp_row[2] and exp_row[3] else None

            return {
                "name": exp_row[0],
                "status": exp_row[1],
                "start_time": exp_row[2],
                "end_time": exp_row[3],
                "current_step": exp_row[4],
                "duration_seconds": duration,
                "checkpoint_counts": checkpoint_counts,
                "metrics_summary": metrics_summary,
            }

    def _row_to_checkpoint(self, row) -> Checkpoint:
        """Optimized row-to-checkpoint conversion - eliminate duplication"""
        import json

        return Checkpoint(
            id=row[0],
            experiment_id=row[1],
            epoch=row[2],
            step=row[3] or 0,
            checkpoint_type=row[4],
            file_path=row[5],
            file_size=row[6],
            checksum=row[7],
            model_name=row[8],
            loss=row[9],
            val_loss=row[10],
            notes=row[11],
            is_best_loss=bool(row[12]) if row[12] is not None else False,
            is_best_val_loss=bool(row[13]) if row[13] is not None else False,
            is_best_metric=bool(row[14]) if row[14] is not None else False,
            metrics=json.loads(row[15]) if row[15] else {},
            metadata=json.loads(row[16]) if row[16] else {},
            created_at=row[17],
        )
