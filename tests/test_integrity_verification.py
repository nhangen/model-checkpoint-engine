# Tests for integrity verification system

import hashlib
import os
import shutil
import tempfile

import pytest
from model_checkpoint.database.enhanced_connection import EnhancedDatabaseConnection
from model_checkpoint.database.models import Checkpoint
from model_checkpoint.integrity import (
    CheckpointVerifier,
    ChecksumCalculator,
    IntegrityTracker,
)


class TestChecksumCalculator:
    # Test checksum calculation functionality

    @pytest.fixture
    def temp_dir(self):
        # Create temporary directory for tests
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_file(self, temp_dir):
        # Create test file with known content
        file_path = os.path.join(temp_dir, "test_file.txt")
        content = b"Hello, World! This is a test file for checksum calculation."
        with open(file_path, 'wb') as f:
            f.write(content)
        return file_path, content

    @pytest.fixture
    def calculator(self):
        # Create checksum calculator
        return ChecksumCalculator(algorithm='sha256')

    def test_file_checksum_calculation(self, calculator, test_file):
        # Test file checksum calculation
        file_path, content = test_file

        # Calculate checksum
        checksum = calculator.calculate_file_checksum(file_path)

        # Verify against manual calculation
        expected_checksum = hashlib.sha256(content).hexdigest()
        assert checksum == expected_checksum

    def test_data_checksum_calculation(self, calculator):
        # Test raw data checksum calculation
        data = b"Test data for checksum"
        checksum = calculator.calculate_data_checksum(data)

        # Verify against manual calculation
        expected_checksum = hashlib.sha256(data).hexdigest()
        assert checksum == expected_checksum

    def test_file_verification(self, calculator, test_file):
        # Test file checksum verification
        file_path, content = test_file

        # Calculate correct checksum
        correct_checksum = calculator.calculate_file_checksum(file_path)

        # Verify with correct checksum
        assert calculator.verify_file_checksum(file_path, correct_checksum) is True

        # Verify with incorrect checksum
        wrong_checksum = "0" * 64  # Wrong checksum
        assert calculator.verify_file_checksum(file_path, wrong_checksum) is False

    def test_checksum_with_metadata(self, calculator, test_file):
        # Test checksum calculation with metadata
        file_path, content = test_file

        metadata = calculator.calculate_with_metadata(file_path)

        assert 'checksum' in metadata
        assert 'algorithm' in metadata
        assert 'file_size' in metadata
        assert 'modification_time' in metadata
        assert 'calculation_time' in metadata
        assert 'file_path' in metadata

        assert metadata['algorithm'] == 'sha256'
        assert metadata['file_size'] == len(content)
        assert metadata['file_path'] == file_path

    def test_comprehensive_verification(self, calculator, test_file):
        # Test comprehensive file verification
        file_path, content = test_file

        # Get expected values
        expected_checksum = calculator.calculate_file_checksum(file_path)
        expected_size = len(content)

        # Verify with all checks
        result = calculator.verify_with_metadata(
            file_path, expected_checksum, expected_size
        )

        assert result['file_exists'] is True
        assert result['checksum_match'] is True
        assert result['size_match'] is True
        assert result['actual_checksum'] == expected_checksum
        assert result['actual_size'] == expected_size

    def test_missing_file_verification(self, calculator):
        # Test verification of missing file
        missing_file = "/path/to/nonexistent/file.txt"

        result = calculator.verify_with_metadata(
            missing_file, "dummy_checksum", 100
        )

        assert result['file_exists'] is False
        assert result['checksum_match'] is False

    def test_different_algorithms(self, temp_dir, test_file):
        # Test different hash algorithms
        file_path, content = test_file

        # Test MD5
        md5_calc = ChecksumCalculator(algorithm='md5')
        md5_checksum = md5_calc.calculate_file_checksum(file_path)
        expected_md5 = hashlib.md5(content).hexdigest()
        assert md5_checksum == expected_md5

        # Test SHA1
        sha1_calc = ChecksumCalculator(algorithm='sha1')
        sha1_checksum = sha1_calc.calculate_file_checksum(file_path)
        expected_sha1 = hashlib.sha1(content).hexdigest()
        assert sha1_checksum == expected_sha1


class TestIntegrityTracker:
    # Test integrity tracking functionality

    @pytest.fixture
    def temp_dir(self):
        # Create temporary directory for tests
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def tracker(self, temp_dir):
        # Create integrity tracker
        tracker_file = os.path.join(temp_dir, "integrity_tracker.json")
        return IntegrityTracker(tracker_file)

    @pytest.fixture
    def test_files(self, temp_dir):
        # Create multiple test files
        files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"test_file_{i}.txt")
            content = f"Test content for file {i}".encode()
            with open(file_path, 'wb') as f:
                f.write(content)
            files.append(file_path)
        return files

    def test_add_file_tracking(self, tracker, test_files):
        # Test adding files to integrity tracking
        file_path = test_files[0]

        # Add file to tracking
        record = tracker.add_file(file_path)

        assert 'checksum' in record
        assert 'file_size' in record
        assert 'modification_time' in record
        assert 'tracked_at' in record
        assert record['file_path'] == os.path.abspath(file_path)

    def test_verify_tracked_file(self, tracker, test_files):
        # Test verifying tracked file
        file_path = test_files[0]

        # Add file to tracking
        tracker.add_file(file_path)

        # Verify file
        result = tracker.verify_file(file_path)

        assert result['status'] == 'verified'
        assert result['file_exists'] is True
        assert result['checksum_match'] is True

    def test_verify_modified_file(self, tracker, test_files, temp_dir):
        # Test verifying file after modification
        file_path = test_files[0]

        # Add file to tracking
        tracker.add_file(file_path)

        # Modify file
        with open(file_path, 'a') as f:
            f.write(" MODIFIED")

        # Verify file (should detect modification)
        result = tracker.verify_file(file_path)

        assert result['status'] in ['checksum_mismatch', 'size_mismatch']

    def test_verify_all_files(self, tracker, test_files):
        # Test verifying all tracked files
        # Add multiple files to tracking
        for file_path in test_files:
            tracker.add_file(file_path)

        # Verify all files
        results = tracker.verify_all()

        assert results['total_files'] == len(test_files)
        assert results['verified'] == len(test_files)
        assert results['corrupted'] == 0
        assert results['missing'] == 0

        # Check individual results
        for file_path in test_files:
            abs_path = os.path.abspath(file_path)
            assert abs_path in results['details']
            assert results['details'][abs_path]['status'] == 'verified'

    def test_missing_file_detection(self, tracker, test_files):
        # Test detection of missing files
        file_path = test_files[0]

        # Add file to tracking
        tracker.add_file(file_path)

        # Remove file
        os.unlink(file_path)

        # Verify file (should detect missing)
        result = tracker.verify_file(file_path)

        assert result['status'] == 'file_missing'
        assert result['file_exists'] is False

    def test_cleanup_missing_files(self, tracker, test_files):
        # Test cleanup of missing file records
        # Add files to tracking
        for file_path in test_files:
            tracker.add_file(file_path)

        # Remove some files
        files_to_remove = test_files[:2]
        for file_path in files_to_remove:
            os.unlink(file_path)

        # Cleanup missing files
        removed_count = tracker.cleanup_missing()

        assert removed_count == len(files_to_remove)

        # Verify remaining files
        results = tracker.verify_all()
        assert results['total_files'] == len(test_files) - len(files_to_remove)

    def test_tracker_statistics(self, tracker, test_files):
        # Test integrity tracker statistics
        # Add files to tracking
        for file_path in test_files:
            tracker.add_file(file_path)

        # Get statistics
        stats = tracker.get_statistics()

        assert stats['total_files'] == len(test_files)
        assert 'total_size_bytes' in stats
        assert 'average_size_bytes' in stats
        assert stats['total_files'] > 0


class TestCheckpointVerifier:
    # Test checkpoint verification system

    @pytest.fixture
    def temp_dir(self):
        # Create temporary directory for tests
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def db_connection(self, temp_dir):
        # Create test database connection
        db_path = os.path.join(temp_dir, "test_verification.db")
        return EnhancedDatabaseConnection(f"sqlite:///{db_path}")

    @pytest.fixture
    def verifier(self, db_connection):
        # Create checkpoint verifier
        return CheckpointVerifier(db_connection)

    @pytest.fixture
    def test_checkpoint(self, temp_dir, db_connection):
        # Create test checkpoint with file and database record
        import torch

        # Create checkpoint file
        checkpoint_data = {
            'model_state_dict': {'layer.weight': torch.randn(10, 5)},
            'optimizer_state_dict': {'param_groups': []},
            'epoch': 10,
            'metrics': {'loss': 0.5, 'accuracy': 0.95}
        }

        file_path = os.path.join(temp_dir, "test_checkpoint.pth")
        torch.save(checkpoint_data, file_path)

        # Calculate checksum
        calculator = ChecksumCalculator()
        checksum = calculator.calculate_file_checksum(file_path)
        file_size = os.path.getsize(file_path)

        # Create database record
        checkpoint_record = Checkpoint(
            id="test_checkpoint_001",
            experiment_id="test_exp_001",
            epoch=10,
            step=1000,
            checkpoint_type="manual",
            file_path=file_path,
            file_size=file_size,
            checksum=checksum,
            loss=0.5,
            metrics={'loss': 0.5, 'accuracy': 0.95}
        )

        db_connection.save_checkpoint(checkpoint_record)

        return checkpoint_record, file_path

    def test_verify_valid_checkpoint(self, verifier, test_checkpoint):
        # Test verification of valid checkpoint
        checkpoint_record, file_path = test_checkpoint

        result = verifier.verify_checkpoint(checkpoint_record.id)

        assert result['status'] == 'verified'
        assert result['checkpoint_id'] == checkpoint_record.id
        assert all(result['checks'].values())  # All checks should pass
        assert len(result['errors']) == 0

    def test_verify_missing_file(self, verifier, test_checkpoint):
        # Test verification of checkpoint with missing file
        checkpoint_record, file_path = test_checkpoint

        # Remove the file
        os.unlink(file_path)

        result = verifier.verify_checkpoint(checkpoint_record.id)

        assert result['status'] == 'file_missing'
        assert result['checks']['file_exists'] is False
        assert len(result['errors']) > 0

    def test_verify_corrupted_checksum(self, verifier, test_checkpoint):
        # Test verification of checkpoint with corrupted file
        checkpoint_record, file_path = test_checkpoint

        # Corrupt the file by appending data
        with open(file_path, 'ab') as f:
            f.write(b"CORRUPTED DATA")

        result = verifier.verify_checkpoint(checkpoint_record.id)

        assert result['status'] in ['checksum_mismatch', 'size_mismatch']
        assert not (result['checks']['checksum_match'] and result['checks']['size_match'])

    def test_verify_experiment_checkpoints(self, verifier, db_connection, temp_dir):
        # Test verification of all checkpoints in an experiment
        import torch

        experiment_id = "test_exp_002"
        checkpoint_ids = []

        # Create multiple checkpoints
        for i in range(3):
            checkpoint_data = {
                'model_state_dict': {'layer.weight': torch.randn(5, 3)},
                'epoch': i + 1,
                'metrics': {'loss': 1.0 / (i + 1)}
            }

            file_path = os.path.join(temp_dir, f"checkpoint_{i}.pth")
            torch.save(checkpoint_data, file_path)

            # Calculate metadata
            calculator = ChecksumCalculator()
            checksum = calculator.calculate_file_checksum(file_path)
            file_size = os.path.getsize(file_path)

            # Create database record
            checkpoint_id = f"checkpoint_{i:03d}"
            checkpoint_record = Checkpoint(
                id=checkpoint_id,
                experiment_id=experiment_id,
                epoch=i + 1,
                file_path=file_path,
                file_size=file_size,
                checksum=checksum,
                loss=1.0 / (i + 1)
            )

            db_connection.save_checkpoint(checkpoint_record)
            checkpoint_ids.append(checkpoint_id)

        # Corrupt one checkpoint
        corrupt_file = os.path.join(temp_dir, "checkpoint_1.pth")
        with open(corrupt_file, 'ab') as f:
            f.write(b"CORRUPTED")

        # Verify all checkpoints
        results = verifier.verify_experiment_checkpoints(experiment_id)

        assert results['total_checkpoints'] == 3
        assert results['verified'] == 2  # Two should be verified
        assert results['failed'] == 1   # One should fail

        # Check individual results
        assert len(results['checkpoint_results']) == 3
        for checkpoint_id, result in results['checkpoint_results'].items():
            if checkpoint_id == 'checkpoint_001':  # The corrupted one
                assert result['status'] in ['checksum_mismatch', 'size_mismatch', 'corrupted']
            else:
                assert result['status'] == 'verified'

    def test_checkpoint_backup_creation(self, verifier, test_checkpoint, temp_dir):
        # Test creating checkpoint backups
        checkpoint_record, file_path = test_checkpoint

        # Create backup
        backup_result = verifier.create_checkpoint_backup(
            checkpoint_record.id,
            backup_dir=os.path.join(temp_dir, "backups")
        )

        assert backup_result['success'] is True
        assert os.path.exists(backup_result['backup_path'])
        assert 'backup_checksum' in backup_result

        # Verify backup file exists and is valid
        backup_path = backup_result['backup_path']
        assert os.path.exists(backup_path)

        # Verify backup can be loaded
        import torch
        backup_data = torch.load(backup_path, map_location='cpu')
        assert 'model_state_dict' in backup_data
        assert backup_data['epoch'] == 10

    def test_nonexistent_checkpoint_verification(self, verifier):
        # Test verification of non-existent checkpoint
        result = verifier.verify_checkpoint("nonexistent_checkpoint")

        assert result['status'] == 'database_record_missing'
        assert result['checks']['database_record'] is False
        assert len(result['errors']) > 0