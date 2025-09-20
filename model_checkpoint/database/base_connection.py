"""Base database connection with shared functionality"""

import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from .models import Experiment, Metric, Checkpoint


class BaseDatabaseConnection:
    """Base SQLite database connection with core functionality"""

    def __init__(self, database_url: str = "sqlite:///experiments.db"):
        """Initialize base database connection"""
        self.db_path = self._extract_db_path(database_url)
        self.logger = logging.getLogger(__name__)
        self._init_tables()

    @staticmethod
    def _extract_db_path(database_url: str) -> str:
        """Extract database path from URL - optimized single method"""
        return database_url[10:] if database_url.startswith("sqlite:///") else database_url

    @contextmanager
    def _get_connection(self):
        """Get optimized database connection with context management"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable performance optimizations
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 10000")
            conn.execute("PRAGMA temp_store = memory")
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        """Initialize database tables with enhanced schema"""
        with self._get_connection() as conn:
            self._create_experiments_table(conn)
            self._create_metrics_table(conn)
            self._create_checkpoints_table(conn)
            conn.commit()

    def _create_experiments_table(self, conn):
        """Create experiments table - centralized definition"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                project_name TEXT,
                status TEXT DEFAULT 'running',
                start_time REAL,
                end_time REAL,
                tags TEXT,  -- JSON array
                config TEXT,  -- JSON object
                step INTEGER DEFAULT 0
            )
        """)

    def _create_metrics_table(self, conn):
        """Create metrics table - centralized definition"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                step INTEGER,
                timestamp REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)

    def _create_checkpoints_table(self, conn):
        """Create checkpoints table - centralized definition"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                experiment_id TEXT,
                epoch INTEGER,
                step INTEGER DEFAULT 0,
                checkpoint_type TEXT DEFAULT 'manual',
                file_path TEXT,
                file_size INTEGER,
                checksum TEXT,
                model_name TEXT,
                loss REAL,
                val_loss REAL,
                notes TEXT,
                is_best_loss BOOLEAN DEFAULT FALSE,
                is_best_val_loss BOOLEAN DEFAULT FALSE,
                is_best_metric BOOLEAN DEFAULT FALSE,
                metrics TEXT,  -- JSON object
                metadata TEXT,  -- JSON object
                created_at REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)

    # Core CRUD operations - shared between all connection types
    def save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to database - optimized single implementation"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (id, name, project_name, status, start_time, end_time, tags, config, step)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id, experiment.name, experiment.project_name,
                experiment.status, experiment.start_time, experiment.end_time,
                json.dumps(experiment.tags), json.dumps(experiment.config),
                experiment.step
            ))
            conn.commit()

    def update_experiment(self, experiment: Experiment) -> None:
        """Update existing experiment - optimized to reuse save_experiment"""
        self.save_experiment(experiment)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID - optimized with proper field handling"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, name, project_name, status, start_time, end_time, tags, config, step
                FROM experiments WHERE id = ?
            """, (experiment_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return Experiment(
                id=row[0], name=row[1], project_name=row[2], status=row[3],
                start_time=row[4], end_time=row[5],
                tags=json.loads(row[6]) if row[6] else [],
                config=json.loads(row[7]) if row[7] else {},
                step=row[8] if row[8] is not None else 0
            )

    def save_metric(self, metric: Metric) -> None:
        """Save metric to database - optimized single implementation"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO metrics
                (experiment_id, metric_name, metric_value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric.experiment_id, metric.metric_name, metric.metric_value,
                metric.step, metric.timestamp
            ))
            conn.commit()

    def get_metrics(self, experiment_id: str, metric_name: Optional[str] = None,
                   step_range: Optional[tuple] = None) -> List[Dict]:
        """Get metrics with enhanced filtering - optimized query building"""
        with self._get_connection() as conn:
            query_parts = ["SELECT metric_name, metric_value, step, timestamp FROM metrics WHERE experiment_id = ?"]
            params = [experiment_id]

            if metric_name:
                query_parts.append("AND metric_name = ?")
                params.append(metric_name)

            if step_range:
                query_parts.append("AND step BETWEEN ? AND ?")
                params.extend(step_range)

            query_parts.append("ORDER BY timestamp, step")
            query = " ".join(query_parts)

            cursor = conn.execute(query, params)
            return [
                {
                    'metric_name': row[0], 'metric_value': row[1],
                    'step': row[2], 'timestamp': row[3]
                }
                for row in cursor.fetchall()
            ]

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint with enhanced metadata - optimized single implementation"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO checkpoints
                (id, experiment_id, epoch, step, checkpoint_type, file_path, file_size,
                 checksum, model_name, loss, val_loss, notes, is_best_loss,
                 is_best_val_loss, is_best_metric, metrics, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint.id, checkpoint.experiment_id, checkpoint.epoch, checkpoint.step,
                checkpoint.checkpoint_type, checkpoint.file_path, checkpoint.file_size,
                checkpoint.checksum, checkpoint.model_name, checkpoint.loss, checkpoint.val_loss,
                checkpoint.notes, checkpoint.is_best_loss, checkpoint.is_best_val_loss,
                checkpoint.is_best_metric, json.dumps(checkpoint.metrics),
                json.dumps(checkpoint.metadata), checkpoint.created_at
            ))
            conn.commit()

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint by ID with all enhanced fields - optimized"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, experiment_id, epoch, step, checkpoint_type, file_path,
                       file_size, checksum, model_name, loss, val_loss, notes,
                       is_best_loss, is_best_val_loss, is_best_metric,
                       metrics, metadata, created_at
                FROM checkpoints WHERE id = ?
            """, (checkpoint_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return Checkpoint(
                id=row[0], experiment_id=row[1], epoch=row[2], step=row[3] or 0,
                checkpoint_type=row[4], file_path=row[5], file_size=row[6],
                checksum=row[7], model_name=row[8], loss=row[9], val_loss=row[10],
                notes=row[11], is_best_loss=bool(row[12]) if row[12] is not None else False,
                is_best_val_loss=bool(row[13]) if row[13] is not None else False,
                is_best_metric=bool(row[14]) if row[14] is not None else False,
                metrics=json.loads(row[15]) if row[15] else {},
                metadata=json.loads(row[16]) if row[16] else {},
                created_at=row[17]
            )