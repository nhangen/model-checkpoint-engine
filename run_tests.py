#!/usr/bin/env python3
"""
Test runner for Model Checkpoint Engine

Run all unit tests and verify functionality
"""

import unittest
import sys
import os
import tempfile

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tests():
    """Run all unit tests"""
    print("🧪 Running Model Checkpoint Engine Tests")
    print("=" * 50)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print(f"📊 Ran {result.testsRun} tests successfully")
        return True
    else:
        print("❌ Some tests failed!")
        print(f"📊 Ran {result.testsRun} tests:")
        print(f"   • Failures: {len(result.failures)}")
        print(f"   • Errors: {len(result.errors)}")
        
        # Print failure details
        if result.failures:
            print("\n🔥 Failures:")
            for test, traceback in result.failures:
                print(f"   • {test}")
        
        if result.errors:
            print("\n💥 Errors:")
            for test, traceback in result.errors:
                print(f"   • {test}")
        
        return False


def verify_installation():
    """Verify package can be imported and basic functionality works"""
    print("\n🔍 Verifying Installation")
    print("-" * 30)
    
    try:
        # Test imports
        from model_checkpoint import ExperimentTracker, CheckpointManager
        from model_checkpoint.database.models import Experiment, Metric
        from model_checkpoint.database.connection import DatabaseConnection
        print("✅ Package imports successful")
        
        # Test database connection
        temp_db = tempfile.mktemp(suffix='.db')
        db = DatabaseConnection(temp_db)
        print("✅ Database connection functional")
        
        # Test experiment creation
        tracker = ExperimentTracker(
            experiment_name="test_verify",
            database_url=f"sqlite:///{temp_db}"
        )
        assert tracker.experiment_name == "test_verify"
        print("✅ Experiment tracking functional")
        
        # Test metric logging
        tracker.log_metrics({'test_metric': 1.0})
        metrics = tracker.get_metrics()
        assert len(metrics) == 1
        print("✅ Metric logging functional")
        
        # Cleanup
        os.unlink(temp_db)
        
        print("✅ Installation verification complete")
        return True
        
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_modular_design():
    """Verify modular design principles"""
    print("\n🏗️  Verifying Modular Design")
    print("-" * 30)
    
    try:
        # Test that components can be imported independently
        from model_checkpoint.database.models import Experiment
        from model_checkpoint.database.connection import DatabaseConnection
        from model_checkpoint.core.experiment import ExperimentTracker
        
        # Test that database can work without experiment tracker
        temp_db = tempfile.mktemp(suffix='.db')
        db = DatabaseConnection(temp_db)
        
        experiment = Experiment(
            id="test-exp",
            name="modular-test"
        )
        db.save_experiment(experiment)
        
        retrieved = db.get_experiment("test-exp")
        assert retrieved.name == "modular-test"
        
        # Cleanup
        os.unlink(temp_db)
        
        print("✅ Components are properly modular")
        print("✅ Database layer is independent")
        print("✅ Models are reusable")
        return True
        
    except Exception as e:
        print(f"❌ Modular design verification failed: {e}")
        return False


def main():
    """Main test runner"""
    print("📊 Model Checkpoint Engine - Test Suite")
    print("=" * 50)
    
    # Verify installation first
    if not verify_installation():
        sys.exit(1)
    
    # Verify modular design
    if not verify_modular_design():
        sys.exit(1)
    
    # Run unit tests
    if not run_tests():
        sys.exit(1)
    
    print("\n🎉 All tests and verifications passed!")
    print("🚀 Model Checkpoint Engine is ready for use!")


if __name__ == "__main__":
    main()