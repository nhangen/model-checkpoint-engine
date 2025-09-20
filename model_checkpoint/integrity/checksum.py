"""Checksum calculation and verification for file integrity"""

import os
import time
from typing import Optional, Dict, Any
from pathlib import Path

from ..utils.checksum import calculate_file_checksum, calculate_data_checksum


class ChecksumCalculator:
    """Efficient checksum calculation - now uses shared optimized utilities"""

    def __init__(self, algorithm: str = 'sha256', chunk_size: int = 65536):
        """
        Initialize checksum calculator - optimized with larger default chunk size

        Args:
            algorithm: Hash algorithm ('sha256', 'md5', 'sha1')
            chunk_size: Chunk size for reading large files (64KB for optimal performance)
        """
        import hashlib
        self.algorithm = algorithm.lower()
        self.chunk_size = chunk_size

        # Validate algorithm
        if self.algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Algorithm {algorithm} not available. Available: {hashlib.algorithms_available}")

    def calculate_file_checksum(self, file_path: str) -> str:
        """Calculate checksum for a file - uses shared optimized utility"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return calculate_file_checksum(file_path, self.algorithm, self.chunk_size)

    def calculate_data_checksum(self, data: bytes) -> str:
        """Calculate checksum for raw data - uses shared optimized utility"""
        return calculate_data_checksum(data, self.algorithm)

    def verify_file_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """
        Verify file checksum against expected value

        Args:
            file_path: Path to file
            expected_checksum: Expected checksum value

        Returns:
            True if checksums match
        """
        try:
            actual_checksum = self.calculate_file_checksum(file_path)
            return actual_checksum.lower() == expected_checksum.lower()
        except FileNotFoundError:
            return False

    def calculate_with_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Calculate checksum with additional metadata

        Args:
            file_path: Path to file

        Returns:
            Dictionary with checksum and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        start_time = time.time()

        # Calculate checksum
        checksum = self.calculate_file_checksum(file_path)

        # Get file metadata
        stat = os.stat(file_path)

        return {
            'checksum': checksum,
            'algorithm': self.algorithm,
            'file_size': stat.st_size,
            'modification_time': stat.st_mtime,
            'calculation_time': time.time() - start_time,
            'file_path': file_path
        }

    def verify_with_metadata(self, file_path: str, expected_checksum: str,
                           expected_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Verify file with comprehensive checks

        Args:
            file_path: Path to file
            expected_checksum: Expected checksum
            expected_size: Expected file size (optional)

        Returns:
            Verification results
        """
        result = {
            'file_exists': False,
            'checksum_match': False,
            'size_match': None,
            'file_path': file_path,
            'expected_checksum': expected_checksum,
            'actual_checksum': None,
            'expected_size': expected_size,
            'actual_size': None,
            'verification_time': None
        }

        start_time = time.time()

        try:
            if not os.path.exists(file_path):
                return result

            result['file_exists'] = True

            # Get file size
            actual_size = os.path.getsize(file_path)
            result['actual_size'] = actual_size

            # Check size if expected
            if expected_size is not None:
                result['size_match'] = actual_size == expected_size
                if not result['size_match']:
                    # Size mismatch - file is likely corrupted, skip checksum calculation
                    return result

            # Calculate and verify checksum
            actual_checksum = self.calculate_file_checksum(file_path)
            result['actual_checksum'] = actual_checksum
            result['checksum_match'] = actual_checksum.lower() == expected_checksum.lower()

        except Exception as e:
            result['error'] = str(e)

        finally:
            result['verification_time'] = time.time() - start_time

        return result


class IntegrityTracker:
    """Track and manage file integrity over time"""

    def __init__(self, tracker_file: Optional[str] = None):
        """
        Initialize integrity tracker

        Args:
            tracker_file: File to store integrity records (JSON)
        """
        self.tracker_file = tracker_file or '.integrity_tracker.json'
        self.calculator = ChecksumCalculator()
        self._load_records()

    def _load_records(self) -> None:
        """Load existing integrity records"""
        import json

        self.records = {}
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    self.records = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.records = {}

    def _save_records(self) -> None:
        """Save integrity records to file"""
        import json

        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.records, f, indent=2)
        except IOError:
            pass  # Fail silently if unable to save

    def add_file(self, file_path: str, force_recalculate: bool = False) -> Dict[str, Any]:
        """
        Add file to integrity tracking

        Args:
            file_path: Path to file
            force_recalculate: Recalculate even if already tracked

        Returns:
            File integrity record
        """
        abs_path = os.path.abspath(file_path)

        # Check if already tracked and up to date
        if not force_recalculate and abs_path in self.records:
            record = self.records[abs_path]
            stat = os.stat(abs_path)
            if record.get('modification_time') == stat.st_mtime:
                return record

        # Calculate integrity metadata
        metadata = self.calculator.calculate_with_metadata(abs_path)
        metadata['tracked_at'] = time.time()

        # Store record
        self.records[abs_path] = metadata
        self._save_records()

        return metadata

    def verify_file(self, file_path: str) -> Dict[str, Any]:
        """
        Verify file integrity against tracked record

        Args:
            file_path: Path to file

        Returns:
            Verification results
        """
        abs_path = os.path.abspath(file_path)

        if abs_path not in self.records:
            return {
                'status': 'not_tracked',
                'message': 'File is not being tracked for integrity'
            }

        record = self.records[abs_path]
        result = self.calculator.verify_with_metadata(
            abs_path,
            record['checksum'],
            record.get('file_size')
        )

        # Determine overall status
        if not result['file_exists']:
            status = 'file_missing'
        elif result.get('size_match') is False:
            status = 'size_mismatch'
        elif not result['checksum_match']:
            status = 'checksum_mismatch'
        else:
            status = 'verified'

        result['status'] = status
        result['tracked_record'] = record

        return result

    def verify_all(self) -> Dict[str, Any]:
        """
        Verify all tracked files

        Returns:
            Summary of all verification results
        """
        results = {
            'total_files': len(self.records),
            'verified': 0,
            'corrupted': 0,
            'missing': 0,
            'errors': 0,
            'details': {}
        }

        for file_path in self.records:
            try:
                result = self.verify_file(file_path)
                results['details'][file_path] = result

                if result['status'] == 'verified':
                    results['verified'] += 1
                elif result['status'] == 'file_missing':
                    results['missing'] += 1
                else:
                    results['corrupted'] += 1

            except Exception as e:
                results['errors'] += 1
                results['details'][file_path] = {
                    'status': 'error',
                    'error': str(e)
                }

        return results

    def remove_file(self, file_path: str) -> bool:
        """
        Remove file from integrity tracking

        Args:
            file_path: Path to file

        Returns:
            True if file was tracked and removed
        """
        abs_path = os.path.abspath(file_path)
        if abs_path in self.records:
            del self.records[abs_path]
            self._save_records()
            return True
        return False

    def cleanup_missing(self) -> int:
        """
        Remove records for files that no longer exist

        Returns:
            Number of records removed
        """
        missing_files = []
        for file_path in self.records:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        for file_path in missing_files:
            del self.records[file_path]

        if missing_files:
            self._save_records()

        return len(missing_files)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get integrity tracking statistics

        Returns:
            Statistics about tracked files
        """
        if not self.records:
            return {'total_files': 0}

        file_sizes = [record.get('file_size', 0) for record in self.records.values()]
        total_size = sum(file_sizes)

        return {
            'total_files': len(self.records),
            'total_size_bytes': total_size,
            'average_size_bytes': total_size / len(self.records) if self.records else 0,
            'largest_file_bytes': max(file_sizes) if file_sizes else 0,
            'smallest_file_bytes': min(file_sizes) if file_sizes else 0,
            'tracker_file': self.tracker_file
        }