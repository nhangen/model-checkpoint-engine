"""Tests for the hook system"""

import pytest
import time
from unittest.mock import Mock, MagicMock

from model_checkpoint.hooks import (
    HookManager, HookEvent, HookPriority, BaseHook, HookContext, HookResult
)
from model_checkpoint.hooks.decorators import hook_handler, conditional_hook


class TestHookManager:
    """Test the central hook manager"""

    @pytest.fixture
    def hook_manager(self):
        return HookManager(enable_async=False)  # Disable async for simpler testing

    def test_hook_manager_initialization(self, hook_manager):
        """Test hook manager initializes correctly"""
        assert hook_manager is not None
        assert hook_manager._global_enabled is True
        assert len(hook_manager._hooks) == 0

    def test_register_simple_hook(self, hook_manager):
        """Test registering a simple hook"""
        def test_handler(context: HookContext):
            return {'test': 'success'}

        hook_manager.register_hook(
            "test_hook",
            test_handler,
            [HookEvent.BEFORE_CHECKPOINT_SAVE]
        )

        assert "test_hook" in hook_manager._hook_registry
        assert len(hook_manager._hooks[HookEvent.BEFORE_CHECKPOINT_SAVE]) == 1

    def test_fire_hook(self, hook_manager):
        """Test firing a hook"""
        results = []

        def test_handler(context: HookContext):
            results.append(context.get('test_data'))
            return {'success': True}

        hook_manager.register_hook(
            "test_hook",
            test_handler,
            [HookEvent.BEFORE_CHECKPOINT_SAVE]
        )

        # Fire the hook
        result = hook_manager.fire_hook(
            HookEvent.BEFORE_CHECKPOINT_SAVE,
            data={'test_data': 'hello world'}
        )

        assert result.success
        assert len(results) == 1
        assert results[0] == 'hello world'

    def test_hook_priority_order(self, hook_manager):
        """Test that hooks execute in priority order"""
        execution_order = []

        def high_priority_handler(context):
            execution_order.append('high')
            return True

        def low_priority_handler(context):
            execution_order.append('low')
            return True

        def normal_priority_handler(context):
            execution_order.append('normal')
            return True

        hook_manager.register_hook(
            "low", low_priority_handler, [HookEvent.BEFORE_CHECKPOINT_SAVE],
            priority=HookPriority.LOW
        )
        hook_manager.register_hook(
            "high", high_priority_handler, [HookEvent.BEFORE_CHECKPOINT_SAVE],
            priority=HookPriority.HIGH
        )
        hook_manager.register_hook(
            "normal", normal_priority_handler, [HookEvent.BEFORE_CHECKPOINT_SAVE],
            priority=HookPriority.NORMAL
        )

        hook_manager.fire_hook(HookEvent.BEFORE_CHECKPOINT_SAVE)

        assert execution_order == ['high', 'normal', 'low']

    def test_hook_can_stop_execution(self, hook_manager):
        """Test that a hook can stop execution chain"""
        results = []

        def stopping_handler(context):
            results.append('stopping')
            return {'success': True, 'continue': False}

        def should_not_run_handler(context):
            results.append('should_not_run')
            return True

        hook_manager.register_hook(
            "stopping", stopping_handler, [HookEvent.BEFORE_CHECKPOINT_SAVE],
            priority=HookPriority.HIGH
        )
        hook_manager.register_hook(
            "blocked", should_not_run_handler, [HookEvent.BEFORE_CHECKPOINT_SAVE],
            priority=HookPriority.LOW
        )

        result = hook_manager.fire_hook(HookEvent.BEFORE_CHECKPOINT_SAVE)

        assert not result.success
        assert result.stopped_by == 'stopping'
        assert results == ['stopping']  # Second handler should not run

    def test_unregister_hook(self, hook_manager):
        """Test unregistering hooks"""
        def test_handler(context):
            return True

        hook_manager.register_hook(
            "test_hook", test_handler, [HookEvent.BEFORE_CHECKPOINT_SAVE]
        )

        assert "test_hook" in hook_manager._hook_registry

        hook_manager.unregister_hook("test_hook")

        assert "test_hook" not in hook_manager._hook_registry
        assert len(hook_manager._hooks[HookEvent.BEFORE_CHECKPOINT_SAVE]) == 0


class TestHookContext:
    """Test the hook context object"""

    def test_context_creation(self):
        """Test creating a hook context"""
        context = HookContext(
            event=HookEvent.BEFORE_CHECKPOINT_SAVE,
            data={'key': 'value'},
            checkpoint_id='test-id'
        )

        assert context.event == HookEvent.BEFORE_CHECKPOINT_SAVE
        assert context.get('key') == 'value'
        assert context.checkpoint_id == 'test-id'

    def test_context_copy(self):
        """Test copying a context"""
        original = HookContext(
            event=HookEvent.BEFORE_CHECKPOINT_SAVE,
            data={'key': 'value'},
            checkpoint_id='test-id'
        )

        copied = original.copy()

        assert copied.event == original.event
        assert copied.get('key') == original.get('key')
        assert copied.checkpoint_id == original.checkpoint_id
        assert copied is not original
        assert copied.data is not original.data

    def test_context_updates(self):
        """Test updating context data"""
        context = HookContext(event=HookEvent.BEFORE_CHECKPOINT_SAVE)

        context.set('new_key', 'new_value')
        assert context.get('new_key') == 'new_value'

        context.update({'batch_key': 'batch_value', 'another': 123})
        assert context.get('batch_key') == 'batch_value'
        assert context.get('another') == 123


class TestBaseHook:
    """Test the base hook class"""

    def test_hook_method_discovery(self):
        """Test that hook methods are discovered correctly"""

        class TestHook(BaseHook):
            def on_init(self):
                pass

            def on_before_checkpoint_save(self, context):
                return True

            def on_after_checkpoint_save(self, context):
                return True

            def regular_method(self):
                return "not a hook"

        hook = TestHook()
        methods = hook.get_hook_methods()

        assert 'on_before_checkpoint_save' in methods
        assert 'on_after_checkpoint_save' in methods
        assert 'regular_method' not in methods

        # Check method configuration
        save_config = methods['on_before_checkpoint_save']
        assert HookEvent.BEFORE_CHECKPOINT_SAVE in save_config['events']


class TestHookDecorators:
    """Test hook decorators"""

    def test_hook_handler_decorator(self):
        """Test the hook_handler decorator"""

        @hook_handler([HookEvent.BEFORE_CHECKPOINT_SAVE], priority=HookPriority.HIGH)
        def decorated_handler(context):
            return True

        assert hasattr(decorated_handler, '_hook_events')
        assert decorated_handler._hook_events == [HookEvent.BEFORE_CHECKPOINT_SAVE]
        assert decorated_handler._priority == HookPriority.HIGH

    def test_conditional_hook_decorator(self):
        """Test the conditional_hook decorator"""

        @conditional_hook(lambda ctx: ctx.get('condition') == True)
        def conditional_handler(context):
            return {'executed': True}

        # Should execute when condition is True
        context = HookContext(event=HookEvent.BEFORE_CHECKPOINT_SAVE, data={'condition': True})
        result = conditional_handler(context)
        assert result['executed'] is True

        # Should skip when condition is False
        context = HookContext(event=HookEvent.BEFORE_CHECKPOINT_SAVE, data={'condition': False})
        result = conditional_handler(context)
        assert result['skipped'] is True


class TestHookIntegration:
    """Test hook integration with checkpoint engine components"""

    def test_checkpoint_manager_hooks(self):
        """Test that checkpoint manager fires hooks correctly"""
        from model_checkpoint.checkpoint.enhanced_manager import EnhancedCheckpointManager

        # Create manager with hooks enabled
        manager = EnhancedCheckpointManager(enable_hooks=True)

        # Verify hook manager exists
        assert manager.hook_manager is not None

        # Register a test hook
        hook_called = []

        def test_hook(context):
            hook_called.append(context.checkpoint_id)
            return True

        manager.register_hook("test", test_hook, [HookEvent.BEFORE_CHECKPOINT_SAVE])

        # Verify hook is registered
        hooks = manager.list_hooks()
        assert len(hooks) > 0
        assert any(h['name'] == 'test' for h in hooks)

    def test_metrics_collector_hooks(self):
        """Test that metrics collector fires hooks correctly"""
        from model_checkpoint.analytics.metrics_collector import MetricsCollector

        # Create collector with hooks enabled
        collector = MetricsCollector(enable_hooks=True)

        # Verify hook manager exists
        assert collector.hook_manager is not None

        # Register a test hook
        hook_data = []

        def metric_hook(context):
            hook_data.append({
                'metric_name': context.get('metric_name'),
                'value': context.get('value')
            })
            return True

        collector.hook_manager.register_hook(
            "metric_test",
            metric_hook,
            [HookEvent.BEFORE_METRIC_COLLECTION]
        )

        # Collect a metric (should trigger hook)
        collector.collect_metric("test_metric", 0.5)

        # Verify hook was called
        assert len(hook_data) == 1
        assert hook_data[0]['metric_name'] == "test_metric"
        assert hook_data[0]['value'] == 0.5


class TestHookPerformance:
    """Test hook system performance"""

    def test_hook_performance_tracking(self):
        """Test that hook performance is tracked"""
        hook_manager = HookManager(enable_async=False)

        def slow_handler(context):
            time.sleep(0.01)  # 10ms
            return True

        hook_manager.register_hook(
            "slow_hook", slow_handler, [HookEvent.BEFORE_CHECKPOINT_SAVE]
        )

        # Fire hook and check performance stats
        hook_manager.fire_hook(HookEvent.BEFORE_CHECKPOINT_SAVE)

        stats = hook_manager.get_performance_stats("slow_hook")
        assert stats['total_calls'] == 1
        assert stats['successful_calls'] == 1
        assert stats['min_time'] >= 0.01  # At least 10ms
        assert stats['avg_time'] >= 0.01

    def test_multiple_hooks_performance(self):
        """Test performance with multiple hooks"""
        hook_manager = HookManager(enable_async=False)

        # Register multiple hooks
        for i in range(10):
            def handler(context, hook_id=i):
                return {'hook_id': hook_id}

            hook_manager.register_hook(
                f"hook_{i}", handler, [HookEvent.BEFORE_CHECKPOINT_SAVE]
            )

        # Fire hooks multiple times
        start_time = time.time()
        for _ in range(100):
            result = hook_manager.fire_hook(HookEvent.BEFORE_CHECKPOINT_SAVE)
            assert result.success

        total_time = time.time() - start_time
        # Should complete 1000 hook executions reasonably quickly
        assert total_time < 1.0  # Less than 1 second


class TestHookErrorHandling:
    """Test hook error handling"""

    def test_hook_error_handling(self):
        """Test that hook errors are handled gracefully"""
        hook_manager = HookManager(enable_async=False)

        def failing_handler(context):
            raise ValueError("Test error")

        def working_handler(context):
            return {'success': True}

        hook_manager.register_hook(
            "failing", failing_handler, [HookEvent.BEFORE_CHECKPOINT_SAVE],
            priority=HookPriority.HIGH
        )
        hook_manager.register_hook(
            "working", working_handler, [HookEvent.BEFORE_CHECKPOINT_SAVE],
            priority=HookPriority.LOW
        )

        # Fire hooks - should handle error gracefully
        result = hook_manager.fire_hook(HookEvent.BEFORE_CHECKPOINT_SAVE)

        # Should have results for both hooks
        assert "failing" in result.results
        assert "working" in result.results

        # Failing hook should have error recorded
        failing_result = result.get_result("failing")
        assert failing_result['success'] is False
        assert "Test error" in failing_result['error']

        # Working hook should succeed
        working_result = result.get_result("working")
        assert working_result['success'] is True


if __name__ == "__main__":
    pytest.main([__file__])