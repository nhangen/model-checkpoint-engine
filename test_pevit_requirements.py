#!/usr/bin/env python3
"""
Test MCE Functionality for PEVIT/Core-Export Requirements

Verifies that model-checkpoint-engine has all the functionality
required by pose-estimation-vit and core-export projects.
"""

import sys
from pathlib import Path


def test_core_imports():
    """Test that all required imports work."""
    print("Testing Core Imports...")
    
    try:
        from model_checkpoint import ExperimentTracker
        print("  ✓ ExperimentTracker")
    except ImportError as e:
        print(f"  ✗ ExperimentTracker: {e}")
        return False
    
    try:
        from model_checkpoint.checkpoint.enhanced_manager import (
            EnhancedCheckpointManager,
        )
        print("  ✓ EnhancedCheckpointManager")
    except ImportError as e:
        print(f"  ✗ EnhancedCheckpointManager: {e}")
        return False
    
    try:
        from model_checkpoint.reporting.html import HTMLReportGenerator
        print("  ✓ HTMLReportGenerator")
    except ImportError as e:
        print(f"  ✗ HTMLReportGenerator: {e}")
        return False
    
    return True


def test_hook_system():
    """Test that hook system is available."""
    print("\nTesting Hook System...")
    
    try:
        from model_checkpoint.hooks.base_hook import BaseHook, HookContext
        print("  ✓ BaseHook, HookContext")
    except ImportError as e:
        print(f"  ✗ BaseHook, HookContext: {e}")
        return False
    
    try:
        from model_checkpoint.hooks.hook_manager import HookEvent
        print("  ✓ HookEvent")
    except ImportError as e:
        print(f"  ✗ HookEvent: {e}")
        return False
    
    try:
        from model_checkpoint.hooks import HookPriority
        print("  ✓ HookPriority")
    except ImportError as e:
        print(f"  ✗ HookPriority: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test that basic functionality works."""
    print("\nTesting Basic Functionality...")
    
    try:
        from model_checkpoint import ExperimentTracker

        # Create tracker
        tracker = ExperimentTracker(
            experiment_name="test",
            project_name="test-project",
            config={"test": True}
        )
        print("  ✓ ExperimentTracker instantiation")
        
        # Check methods exist
        assert hasattr(tracker, 'start_experiment'), "Missing start_experiment"
        assert hasattr(tracker, 'log_step'), "Missing log_step"
        assert hasattr(tracker, 'finish_experiment'), "Missing finish_experiment"
        print("  ✓ ExperimentTracker has required methods")
        
    except Exception as e:
        print(f"  ✗ ExperimentTracker functionality: {e}")
        return False
    
    try:
        from model_checkpoint.hooks.base_hook import BaseHook, HookContext

        # Create test hook
        class TestHook(BaseHook):
            def on_epoch_end(self, context: HookContext):
                pass
        
        hook = TestHook()
        print("  ✓ Can create custom hooks")
        
    except Exception as e:
        print(f"  ✗ Hook creation: {e}")
        return False
    
    return True


def test_no_vit_dependencies():
    """Verify no VIT-specific code leaked through."""
    print("\nTesting No VIT Dependencies...")
    
    # Check that validation module was removed
    try:
        from model_checkpoint.validation import SystemValidator
        print("  ✗ validation module still exists (should be removed)")
        return False
    except ImportError:
        print("  ✓ validation module removed (correct)")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("MCE FUNCTIONALITY TEST FOR PEVIT/CORE-EXPORT")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Hook System", test_hook_system),
        ("Basic Functionality", test_basic_functionality),
        ("No VIT Dependencies", test_no_vit_dependencies),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - MCE has required functionality")
        return 0
    else:
        print("❌ SOME TESTS FAILED - MCE missing required functionality")
        return 1


if __name__ == "__main__":
    sys.exit(main())
