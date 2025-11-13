-- Migration: Add step tracking and enhanced experiment fields
-- Version: 001
-- Created: 2025-09-20

-- Add step column to experiments table for step-level granularity (if not exists)
-- SQLite doesn't support IF NOT EXISTS for columns, so we need to check first

-- Add indices for performance optimization
CREATE INDEX IF NOT EXISTS idx_experiments_step ON experiments(step);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_name);

-- Add step indexing to metrics table
CREATE INDEX IF NOT EXISTS idx_metrics_step ON metrics(step);
CREATE INDEX IF NOT EXISTS idx_metrics_experiment_step ON metrics(experiment_id, step);
CREATE INDEX IF NOT EXISTS idx_metrics_name_step ON metrics(metric_name, step);

-- Add compound indices for complex queries
CREATE INDEX IF NOT EXISTS idx_metrics_experiment_name_step ON metrics(experiment_id, metric_name, step);
