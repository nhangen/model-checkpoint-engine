"""Database migration management system"""

import sqlite3
import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..utils.checksum import calculate_data_checksum


class MigrationManager:
    """Manages database schema migrations with version control"""

    def __init__(self, db_path: str, migrations_dir: Optional[str] = None):
        """
        Initialize migration manager

        Args:
            db_path: Path to SQLite database
            migrations_dir: Directory containing migration files
        """
        self.db_path = db_path
        self.migrations_dir = migrations_dir or str(Path(__file__).parent / "migrations")
        self.logger = logging.getLogger(__name__)

        # Ensure migrations directory exists
        Path(self.migrations_dir).mkdir(parents=True, exist_ok=True)

        # Initialize migration tracking table
        self._init_migration_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with optimized settings"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_migration_table(self) -> None:
        """Initialize migration tracking table"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    executed_at REAL NOT NULL,
                    execution_time_ms INTEGER NOT NULL,
                    checksum TEXT NOT NULL
                )
            """)
            conn.commit()

    def get_current_version(self) -> Optional[str]:
        """Get current database schema version"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT version FROM schema_migrations
                ORDER BY executed_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return row[0] if row else None

    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT version FROM schema_migrations
                ORDER BY executed_at
            """)
            return [row[0] for row in cursor.fetchall()]

    def get_pending_migrations(self) -> List[Dict[str, str]]:
        """Get list of pending migrations"""
        applied = set(self.get_applied_migrations())
        pending = []

        migration_files = self._get_migration_files()
        for migration_file in migration_files:
            version = self._extract_version(migration_file)
            if version not in applied:
                pending.append({
                    'version': version,
                    'filename': migration_file,
                    'path': os.path.join(self.migrations_dir, migration_file)
                })

        return pending

    def _get_migration_files(self) -> List[str]:
        """Get sorted list of migration files"""
        if not os.path.exists(self.migrations_dir):
            return []

        files = [f for f in os.listdir(self.migrations_dir)
                if f.endswith('.sql') and f[0].isdigit()]
        return sorted(files)

    def _extract_version(self, filename: str) -> str:
        """Extract version from migration filename"""
        # Expected format: 001_description.sql
        return filename.split('_')[0]

    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA256 checksum of migration content - uses shared utility"""
        return calculate_data_checksum(content)

    def migrate(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Run pending migrations up to target version

        Args:
            target_version: Stop at this version (None = run all)

        Returns:
            Migration results summary
        """
        pending = self.get_pending_migrations()

        if target_version:
            pending = [m for m in pending if m['version'] <= target_version]

        if not pending:
            self.logger.info("No pending migrations to run")
            return {
                'status': 'success',
                'migrations_run': 0,
                'message': 'No pending migrations'
            }

        results = {
            'status': 'success',
            'migrations_run': 0,
            'executed': [],
            'failed': None,
            'total_time_ms': 0
        }

        start_time = time.time()

        try:
            for migration in pending:
                self.logger.info(f"Running migration {migration['version']}: {migration['filename']}")
                self._run_migration(migration)
                results['executed'].append(migration['version'])
                results['migrations_run'] += 1

        except Exception as e:
            results['status'] = 'failed'
            results['failed'] = migration['version']
            results['error'] = str(e)
            self.logger.error(f"Migration {migration['version']} failed: {e}")
            raise

        results['total_time_ms'] = int((time.time() - start_time) * 1000)
        return results

    def _run_migration(self, migration: Dict[str, str]) -> None:
        """Run a single migration"""
        with open(migration['path'], 'r') as f:
            sql_content = f.read()

        checksum = self._calculate_checksum(sql_content)
        start_time = time.time()

        with self._get_connection() as conn:
            # Execute migration SQL
            for statement in self._split_sql_statements(sql_content):
                if statement.strip():
                    conn.execute(statement)

            # Record migration
            execution_time_ms = int((time.time() - start_time) * 1000)
            conn.execute("""
                INSERT INTO schema_migrations
                (version, filename, executed_at, execution_time_ms, checksum)
                VALUES (?, ?, ?, ?, ?)
            """, (
                migration['version'],
                migration['filename'],
                time.time(),
                execution_time_ms,
                checksum
            ))

            conn.commit()

    def _split_sql_statements(self, sql_content: str) -> List[str]:
        """Split SQL content into individual statements"""
        # Simple split on semicolons (could be enhanced for complex SQL)
        statements = []
        current_statement = []

        for line in sql_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('--'):
                continue

            current_statement.append(line)
            if line.endswith(';'):
                statements.append('\n'.join(current_statement))
                current_statement = []

        return statements

    def rollback(self, target_version: str) -> Dict[str, Any]:
        """
        Rollback to a specific version (if rollback migrations exist)

        Args:
            target_version: Version to rollback to

        Returns:
            Rollback results
        """
        current_version = self.get_current_version()
        if not current_version:
            return {
                'status': 'error',
                'message': 'No migrations to rollback'
            }

        if target_version >= current_version:
            return {
                'status': 'error',
                'message': f'Target version {target_version} is not older than current {current_version}'
            }

        # Note: This is a simplified rollback implementation
        # In production, you'd need proper rollback SQL files
        self.logger.warning("Rollback functionality requires manual rollback SQL files")
        return {
            'status': 'manual',
            'message': 'Rollback requires manual intervention'
        }

    def validate_migrations(self) -> Dict[str, Any]:
        """Validate applied migrations against their checksums"""
        validation_results = {
            'status': 'success',
            'validated': [],
            'mismatched': [],
            'missing_files': []
        }

        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT version, filename, checksum
                FROM schema_migrations
                ORDER BY executed_at
            """)

            for version, filename, stored_checksum in cursor.fetchall():
                migration_path = os.path.join(self.migrations_dir, filename)

                if not os.path.exists(migration_path):
                    validation_results['missing_files'].append(version)
                    validation_results['status'] = 'warning'
                    continue

                with open(migration_path, 'r') as f:
                    current_checksum = self._calculate_checksum(f.read())

                if current_checksum != stored_checksum:
                    validation_results['mismatched'].append(version)
                    validation_results['status'] = 'error'
                else:
                    validation_results['validated'].append(version)

        return validation_results

    def create_migration(self, description: str) -> str:
        """
        Create a new migration file

        Args:
            description: Description of the migration

        Returns:
            Path to created migration file
        """
        # Get next version number
        current_version = self.get_current_version()
        if current_version:
            next_version = f"{int(current_version) + 1:03d}"
        else:
            next_version = "001"

        # Create filename
        clean_description = description.lower().replace(' ', '_').replace('-', '_')
        filename = f"{next_version}_{clean_description}.sql"
        filepath = os.path.join(self.migrations_dir, filename)

        # Create template migration file
        template = f"""-- Migration: {description}
-- Version: {next_version}
-- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}

-- Add your migration SQL here
-- Example:
-- ALTER TABLE experiments ADD COLUMN new_field TEXT;

-- Remember to test your migration thoroughly!
"""

        with open(filepath, 'w') as f:
            f.write(template)

        self.logger.info(f"Created migration file: {filepath}")
        return filepath