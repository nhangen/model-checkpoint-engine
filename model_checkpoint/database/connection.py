"""Database connection and operations"""

import sqlite3
import json
from typing import List, Optional, Dict, Any
from .models import Experiment, Metric, Checkpoint


class DatabaseConnection:
    """Simple SQLite database connection for experiment tracking"""
    
    def __init__(self, database_url: str = "sqlite:///experiments.db"):
        """Initialize database connection"""
        # Extract path from SQLite URL
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url[10:]  # Remove "sqlite:///"
        else:
            self.db_path = database_url
        
        self._init_tables()
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _init_tables(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            # Experiments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    project_name TEXT,
                    status TEXT DEFAULT 'running',
                    start_time REAL,
                    end_time REAL,
                    tags TEXT,  -- JSON array
                    config TEXT  -- JSON object
                )
            """)
            
            # Metrics table
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
            
            # Checkpoints table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    epoch INTEGER,
                    checkpoint_type TEXT DEFAULT 'manual',
                    file_path TEXT,
                    metrics TEXT,  -- JSON object
                    metadata TEXT,  -- JSON object
                    created_at REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            conn.commit()
    
    def save_experiment(self, experiment: Experiment):
        """Save experiment to database"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments 
                (id, name, project_name, status, start_time, end_time, tags, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id,
                experiment.name,
                experiment.project_name,
                experiment.status,
                experiment.start_time,
                experiment.end_time,
                json.dumps(experiment.tags),
                json.dumps(experiment.config)
            ))
            conn.commit()
    
    def update_experiment(self, experiment: Experiment):
        """Update existing experiment"""
        self.save_experiment(experiment)  # Same as save for SQLite
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, name, project_name, status, start_time, end_time, tags, config
                FROM experiments WHERE id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return Experiment(
                id=row[0],
                name=row[1],
                project_name=row[2],
                status=row[3],
                start_time=row[4],
                end_time=row[5],
                tags=json.loads(row[6]) if row[6] else [],
                config=json.loads(row[7]) if row[7] else {}
            )
    
    def save_metric(self, metric: Metric):
        """Save metric to database"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO metrics 
                (experiment_id, metric_name, metric_value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric.experiment_id,
                metric.metric_name,
                metric.metric_value,
                metric.step,
                metric.timestamp
            ))
            conn.commit()
    
    def get_metrics(self, experiment_id: str, metric_name: Optional[str] = None) -> List[Dict]:
        """Get metrics for experiment"""
        with self._get_connection() as conn:
            if metric_name:
                cursor = conn.execute("""
                    SELECT metric_name, metric_value, step, timestamp
                    FROM metrics 
                    WHERE experiment_id = ? AND metric_name = ?
                    ORDER BY timestamp
                """, (experiment_id, metric_name))
            else:
                cursor = conn.execute("""
                    SELECT metric_name, metric_value, step, timestamp
                    FROM metrics 
                    WHERE experiment_id = ?
                    ORDER BY timestamp
                """, (experiment_id,))
            
            return [
                {
                    'metric_name': row[0],
                    'metric_value': row[1],
                    'step': row[2],
                    'timestamp': row[3]
                }
                for row in cursor.fetchall()
            ]
    
    def save_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint to database"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO checkpoints 
                (id, experiment_id, epoch, checkpoint_type, file_path, metrics, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint.id,
                checkpoint.experiment_id,
                checkpoint.epoch,
                checkpoint.checkpoint_type,
                checkpoint.file_path,
                json.dumps(checkpoint.metrics),
                json.dumps(checkpoint.metadata),
                checkpoint.created_at
            ))
            conn.commit()