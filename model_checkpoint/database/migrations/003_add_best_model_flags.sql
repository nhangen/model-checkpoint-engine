-- Migration: Add best model tracking flags
-- Version: 003
-- Created: 2025-09-20

-- Add best model flags to checkpoints table
ALTER TABLE checkpoints ADD COLUMN is_best_loss BOOLEAN DEFAULT FALSE;
ALTER TABLE checkpoints ADD COLUMN is_best_val_loss BOOLEAN DEFAULT FALSE;
ALTER TABLE checkpoints ADD COLUMN is_best_metric BOOLEAN DEFAULT FALSE;

-- Add performance indices for best model queries
CREATE INDEX IF NOT EXISTS idx_checkpoints_best_loss ON checkpoints(is_best_loss);
CREATE INDEX IF NOT EXISTS idx_checkpoints_best_val_loss ON checkpoints(is_best_val_loss);
CREATE INDEX IF NOT EXISTS idx_checkpoints_best_metric ON checkpoints(is_best_metric);

-- Add compound indices for experiment-specific best model queries
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment_best_loss ON checkpoints(experiment_id, is_best_loss);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment_best_val_loss ON checkpoints(experiment_id, is_best_val_loss);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment_best_metric ON checkpoints(experiment_id, is_best_metric);

-- Add index for timestamp-based queries (finding latest best models)
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at);
