-- Replace the time-dependent partial index from migration004. SQLite rejects
-- inserts when an indexed predicate evaluates strftime('now'). Filter by the
-- query's cutoff while retaining an ordinary timestamp index.
DROP INDEX IF EXISTS idx_recent_experiments;
CREATE INDEX idx_recent_experiments ON experiments(start_time);
