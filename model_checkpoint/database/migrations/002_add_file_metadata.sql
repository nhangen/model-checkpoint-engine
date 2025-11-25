-- Migration: Add file metadata and integrity tracking
-- Version: 002
-- Created: 2025-09-20

-- Note: Columns (step, file_size, checksum, model_name, loss, val_loss, notes)
-- already exist in base schema - no ALTER TABLE needed

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
