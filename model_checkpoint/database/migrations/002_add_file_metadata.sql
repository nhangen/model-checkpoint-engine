-- Migration: Add file metadata and integrity tracking
-- Version: 002
-- Created: 2025-09-20

-- Add file metadata columns to checkpoints table
ALTER TABLE checkpoints ADD COLUMN step INTEGER DEFAULT 0;
ALTER TABLE checkpoints ADD COLUMN file_size INTEGER;
ALTER TABLE checkpoints ADD COLUMN checksum TEXT;
ALTER TABLE checkpoints ADD COLUMN model_name TEXT;
ALTER TABLE checkpoints ADD COLUMN loss REAL;
ALTER TABLE checkpoints ADD COLUMN val_loss REAL;
ALTER TABLE checkpoints ADD COLUMN notes TEXT;

-- Add indices for enhanced checkpoint querying
CREATE INDEX IF NOT EXISTS idx_checkpoints_step ON checkpoints(step);
CREATE INDEX IF NOT EXISTS idx_checkpoints_epoch_step ON checkpoints(epoch, step);
CREATE INDEX IF NOT EXISTS idx_checkpoints_checksum ON checkpoints(checksum);
CREATE INDEX IF NOT EXISTS idx_checkpoints_file_size ON checkpoints(file_size);
CREATE INDEX IF NOT EXISTS idx_checkpoints_loss ON checkpoints(loss);
CREATE INDEX IF NOT EXISTS idx_checkpoints_val_loss ON checkpoints(val_loss);

-- Add compound indices for complex checkpoint queries
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment_epoch ON checkpoints(experiment_id, epoch);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment_step ON checkpoints(experiment_id, step);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment_type ON checkpoints(experiment_id, checkpoint_type);