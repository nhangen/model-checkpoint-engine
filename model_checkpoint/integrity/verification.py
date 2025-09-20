"""Comprehensive checkpoint verification and repair system"""

import os
import json
import time
import shutil
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .checksum import ChecksumCalculator, IntegrityTracker
from ..database.enhanced_connection import EnhancedDatabaseConnection


class CheckpointVerifier:
    """Comprehensive checkpoint verification with repair capabilities"""

    def __init__(self, database_connection: EnhancedDatabaseConnection,
                 integrity_tracker: Optional[IntegrityTracker] = None):
        """
        Initialize checkpoint verifier

        Args:
            database_connection: Database connection for checkpoint metadata
            integrity_tracker: Optional integrity tracker for file monitoring
        """
        self.db = database_connection
        self.integrity_tracker = integrity_tracker or IntegrityTracker()
        self.checksum_calculator = ChecksumCalculator()

    def verify_checkpoint(self, checkpoint_id: str, repair_on_failure: bool = False) -> Dict[str, Any]:
        """
        Verify a single checkpoint's integrity

        Args:
            checkpoint_id: Checkpoint ID to verify
            repair_on_failure: Attempt repair if verification fails

        Returns:
            Verification results
        """
        result = {
            'checkpoint_id': checkpoint_id,
            'status': 'unknown',
            'checks': {
                'database_record': False,
                'file_exists': False,
                'checksum_match': False,
                'size_match': False,
                'loadable': False
            },
            'metadata': {},
            'errors': [],
            'repair_attempted': False,
            'repair_successful': False
        }

        start_time = time.time()

        try:
            # 1. Check database record
            checkpoint = self.db.get_checkpoint(checkpoint_id)
            if not checkpoint:
                result['status'] = 'database_record_missing'
                result['errors'].append('Checkpoint record not found in database')
                return result

            result['checks']['database_record'] = True
            result['metadata']['database_record'] = {
                'experiment_id': checkpoint.experiment_id,
                'epoch': checkpoint.epoch,
                'step': checkpoint.step,
                'checkpoint_type': checkpoint.checkpoint_type,
                'file_path': checkpoint.file_path,
                'expected_checksum': checkpoint.checksum,
                'expected_size': checkpoint.file_size
            }

            # 2. Check file existence
            if not checkpoint.file_path or not os.path.exists(checkpoint.file_path):
                result['status'] = 'file_missing'
                result['errors'].append(f'Checkpoint file not found: {checkpoint.file_path}')

                if repair_on_failure:
                    repair_result = self._attempt_file_recovery(checkpoint)
                    result['repair_attempted'] = True
                    result['repair_successful'] = repair_result['success']
                    if repair_result['success']:
                        checkpoint.file_path = repair_result['recovered_path']

                if not result['repair_successful']:
                    return result

            result['checks']['file_exists'] = True

            # 3. Check file size
            actual_size = os.path.getsize(checkpoint.file_path)
            result['metadata']['actual_size'] = actual_size

            if checkpoint.file_size and actual_size != checkpoint.file_size:
                result['status'] = 'size_mismatch'
                result['errors'].append(
                    f'File size mismatch: expected {checkpoint.file_size}, got {actual_size}'
                )
            else:
                result['checks']['size_match'] = True

            # 4. Check checksum
            if checkpoint.checksum:
                verification = self.checksum_calculator.verify_with_metadata(
                    checkpoint.file_path,
                    checkpoint.checksum,
                    checkpoint.file_size
                )

                if verification['checksum_match']:
                    result['checks']['checksum_match'] = True
                else:
                    result['status'] = 'checksum_mismatch'
                    result['errors'].append(
                        f"Checksum mismatch: expected {checkpoint.checksum}, "
                        f"got {verification['actual_checksum']}"
                    )

            # 5. Test loadability
            try:
                # Try to load checkpoint metadata without full loading
                self._test_checkpoint_loadability(checkpoint.file_path)
                result['checks']['loadable'] = True
            except Exception as e:
                result['status'] = 'corrupted'
                result['errors'].append(f'Checkpoint not loadable: {e}')

                if repair_on_failure:
                    repair_result = self._attempt_corruption_repair(checkpoint)
                    result['repair_attempted'] = True
                    result['repair_successful'] = repair_result['success']

            # Determine final status
            if all(result['checks'].values()):
                result['status'] = 'verified'
            elif result['status'] == 'unknown':
                result['status'] = 'partial_failure'

        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(f'Verification error: {e}')

        finally:
            result['verification_time'] = time.time() - start_time

        return result

    def verify_experiment_checkpoints(self, experiment_id: str,
                                    repair_on_failure: bool = False) -> Dict[str, Any]:
        """
        Verify all checkpoints for an experiment

        Args:
            experiment_id: Experiment ID
            repair_on_failure: Attempt repair for failed checkpoints

        Returns:
            Summary of verification results
        """
        checkpoints = self.db.get_checkpoints_by_experiment(experiment_id)

        results = {
            'experiment_id': experiment_id,
            'total_checkpoints': len(checkpoints),
            'verified': 0,
            'failed': 0,
            'repaired': 0,
            'irreparable': 0,
            'checkpoint_results': {},
            'summary': {}
        }

        for checkpoint in checkpoints:
            checkpoint_result = self.verify_checkpoint(
                checkpoint.id,
                repair_on_failure
            )

            results['checkpoint_results'][checkpoint.id] = checkpoint_result

            if checkpoint_result['status'] == 'verified':
                results['verified'] += 1
            else:
                results['failed'] += 1

            if checkpoint_result['repair_attempted']:
                if checkpoint_result['repair_successful']:
                    results['repaired'] += 1
                else:
                    results['irreparable'] += 1

        # Generate summary
        results['summary'] = {
            'integrity_percentage': (results['verified'] / results['total_checkpoints'] * 100)
            if results['total_checkpoints'] > 0 else 0,
            'repair_success_rate': (results['repaired'] / (results['repaired'] + results['irreparable']) * 100)
            if (results['repaired'] + results['irreparable']) > 0 else 0
        }

        return results

    def verify_all_checkpoints(self, repair_on_failure: bool = False,
                             experiment_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Verify all checkpoints in the database

        Args:
            repair_on_failure: Attempt repair for failed checkpoints
            experiment_filter: Only verify checkpoints from these experiments

        Returns:
            Global verification summary
        """
        # This would require a method to get all experiments
        # For now, we'll implement a simplified version
        return {
            'status': 'not_implemented',
            'message': 'Global verification requires experiment enumeration capability'
        }

    def _test_checkpoint_loadability(self, file_path: str) -> bool:
        """
        Test if a checkpoint file can be loaded without errors

        Args:
            file_path: Path to checkpoint file

        Returns:
            True if loadable
        """
        import torch

        try:
            # Quick load test - just check if PyTorch can read the file structure
            checkpoint_data = torch.load(file_path, map_location='cpu')

            # Basic validation
            if not isinstance(checkpoint_data, dict):
                raise ValueError("Checkpoint is not a dictionary")

            # Check for basic structure
            expected_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'metrics']
            if not any(key in checkpoint_data for key in expected_keys):
                raise ValueError("Checkpoint missing expected structure")

            return True

        except Exception:
            return False

    def _attempt_file_recovery(self, checkpoint) -> Dict[str, Any]:
        """
        Attempt to recover a missing checkpoint file

        Args:
            checkpoint: Checkpoint database record

        Returns:
            Recovery attempt results
        """
        recovery_result = {
            'success': False,
            'method': None,
            'recovered_path': None,
            'details': []
        }

        # Try different recovery strategies
        original_path = checkpoint.file_path

        # Strategy 1: Look for backup files
        backup_patterns = [
            original_path + '.backup',
            original_path + '.bak',
            original_path.replace('.pth', '_backup.pth')
        ]

        for backup_path in backup_patterns:
            if os.path.exists(backup_path):
                # Verify backup integrity
                try:
                    if self._test_checkpoint_loadability(backup_path):
                        # Copy backup to original location
                        shutil.copy2(backup_path, original_path)
                        recovery_result.update({
                            'success': True,
                            'method': 'backup_restore',
                            'recovered_path': original_path,
                            'details': [f'Restored from backup: {backup_path}']
                        })
                        return recovery_result
                except Exception as e:
                    recovery_result['details'].append(f'Backup {backup_path} corrupted: {e}')

        # Strategy 2: Look for similar checkpoints in the same experiment
        similar_checkpoints = self.db.get_checkpoints_by_experiment(checkpoint.experiment_id)
        for similar_checkpoint in similar_checkpoints:
            if (similar_checkpoint.id != checkpoint.id and
                similar_checkpoint.epoch == checkpoint.epoch and
                os.path.exists(similar_checkpoint.file_path)):

                recovery_result.update({
                    'success': True,
                    'method': 'similar_checkpoint',
                    'recovered_path': similar_checkpoint.file_path,
                    'details': [f'Using similar checkpoint: {similar_checkpoint.id}']
                })
                return recovery_result

        recovery_result['details'].append('No recovery options found')
        return recovery_result

    def _attempt_corruption_repair(self, checkpoint) -> Dict[str, Any]:
        """
        Attempt to repair a corrupted checkpoint file

        Args:
            checkpoint: Checkpoint database record

        Returns:
            Repair attempt results
        """
        repair_result = {
            'success': False,
            'method': None,
            'details': []
        }

        # For corrupted files, recovery options are limited
        # This is a placeholder for potential repair strategies
        repair_result['details'].append('Corruption repair not yet implemented')

        return repair_result

    def create_checkpoint_backup(self, checkpoint_id: str, backup_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a backup of a checkpoint

        Args:
            checkpoint_id: Checkpoint ID to backup
            backup_dir: Directory for backup (default: same as original + '_backups')

        Returns:
            Backup operation results
        """
        checkpoint = self.db.get_checkpoint(checkpoint_id)
        if not checkpoint:
            return {
                'success': False,
                'error': 'Checkpoint not found in database'
            }

        if not os.path.exists(checkpoint.file_path):
            return {
                'success': False,
                'error': 'Checkpoint file not found'
            }

        try:
            # Determine backup location
            if backup_dir is None:
                original_dir = os.path.dirname(checkpoint.file_path)
                backup_dir = os.path.join(original_dir, '_backups')

            os.makedirs(backup_dir, exist_ok=True)

            # Create backup filename with timestamp
            original_filename = os.path.basename(checkpoint.file_path)
            timestamp = int(time.time())
            backup_filename = f"{timestamp}_{original_filename}"
            backup_path = os.path.join(backup_dir, backup_filename)

            # Copy file
            shutil.copy2(checkpoint.file_path, backup_path)

            # Verify backup
            if self._test_checkpoint_loadability(backup_path):
                # Calculate backup checksum
                backup_checksum = self.checksum_calculator.calculate_file_checksum(backup_path)

                return {
                    'success': True,
                    'backup_path': backup_path,
                    'backup_checksum': backup_checksum,
                    'backup_size': os.path.getsize(backup_path),
                    'original_path': checkpoint.file_path
                }
            else:
                # Remove corrupted backup
                os.unlink(backup_path)
                return {
                    'success': False,
                    'error': 'Backup verification failed'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Backup creation failed: {e}'
            }