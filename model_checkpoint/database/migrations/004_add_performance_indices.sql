-- Migration: Add comprehensive performance indices
-- Version: 004
-- Created: 2025-09-20

-- Advanced compound indices for complex analytical queries
CREATE INDEX IF NOT EXISTS idx_metrics_experiment_timestamp ON metrics(experiment_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_value_timestamp ON metrics(metric_value, timestamp);

-- Partial indices for common filtering patterns
CREATE INDEX IF NOT EXISTS idx_experiments_running
ON experiments(id, start_time) WHERE status = 'running';

CREATE INDEX IF NOT EXISTS idx_experiments_completed
ON experiments(id, end_time) WHERE status = 'completed';

-- Covering indices for dashboard queries (avoid table lookups)
CREATE INDEX IF NOT EXISTS idx_experiments_summary
ON experiments(id, name, status, start_time, end_time);

CREATE INDEX IF NOT EXISTS idx_checkpoints_summary
ON checkpoints(id, experiment_id, epoch, step, checkpoint_type, created_at, loss, val_loss);

-- Indices for metrics aggregation queries
CREATE INDEX IF NOT EXISTS idx_metrics_aggregation
ON metrics(experiment_id, metric_name, metric_value, step, timestamp);

-- Optimize for recent data queries (last 30 days)
CREATE INDEX IF NOT EXISTS idx_recent_experiments
ON experiments(start_time) WHERE start_time > strftime('%s', 'now', '-30 days');

-- Index for checkpoint file path queries (cleanup operations)
CREATE INDEX IF NOT EXISTS idx_checkpoints_file_path ON checkpoints(file_path);
