# ML Model Checkpoint Engine

A high-performance, generic checkpoint management and experiment tracking system for machine learning projects. Built with enterprise-grade scalability and framework-agnostic design.

## Core Features

### Enhanced Checkpoint Management
- **Multi-Backend Storage**: PyTorch, SafeTensors with pluggable architecture
- **Data Integrity**: SHA256 checksum verification with comprehensive tracking  
- **Performance Caching**: LRU with TTL, optimized metadata access
- **Best Model Detection**: Automatic flagging based on configurable metrics
- **Retention Policies**: Configurable cleanup with protected checkpoint types

### Experiment Tracking
- **Database Optimization**: SQLite with connection pooling and WAL mode
- **Metadata Management**: Comprehensive experiment and checkpoint metadata
- **Query Interface**: Advanced filtering and analytics capabilities
- **Statistics**: Real-time performance and usage statistics

### Hook System
- **Event-Driven Architecture**: 15+ hook events for extensibility
- **Priority-Based Execution**: Configurable execution order
- **Error Handling**: Robust failure isolation
- **Async/Sync Support**: Flexible hook execution models

## Architecture Principles

- **Framework Agnostic**: Works with any ML framework (PyTorch, TensorFlow, JAX, etc.)
- **Storage Flexibility**: Pluggable storage backends
- **Thread Safety**: Concurrent operations with proper locking
- **Backward Compatibility**: Seamless integration with existing systems
- **Zero Dependencies**: Minimal external requirements for core functionality

## Quick Start

```python
from model_checkpoint import ExperimentTracker
from model_checkpoint.checkpoint.enhanced_manager import EnhancedCheckpointManager

# Initialize experiment tracking
tracker = ExperimentTracker(
    experiment_name="my_experiment",
    project_name="my_project",
    config={"learning_rate": 0.001, "batch_size": 32}
)

# Initialize checkpoint manager
manager = EnhancedCheckpointManager(
    experiment_tracker=tracker,
    checkpoint_dir="./checkpoints",
    storage_backend="pytorch",
    enable_hooks=True
)

# During training
for epoch in range(epochs):
    # Your training logic here...

    # Save checkpoint with automatic best model detection
    checkpoint_id = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=train_loss,
        val_loss=val_loss,
        metrics={"accuracy": accuracy},
        notes=f"Checkpoint at epoch {epoch}"
    )

    print(f"Saved checkpoint: {checkpoint_id}")

# Load best checkpoint
best_checkpoint = manager.load_checkpoint(
    experiment_id=tracker.experiment_id,
    checkpoint_type="best_val_loss"
)
```

## Installation

```bash
# Install from source
pip install -e .

# Core dependencies are minimal
# - SQLite (built into Python)
# - PyTorch (optional, only if using PyTorch storage backend)
```

## Development Setup

This project uses pre-commit hooks to maintain code quality and consistency.

### Quick Start for Developers

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# (Optional) Run on all files to check current state
pre-commit run --all-files
```

### What Pre-commit Checks

- **Code Formatting**: Black, isort
- **Linting**: Flake8 with plugins (docstrings, bugbear, comprehensions)
- **Type Checking**: MyPy
- **Security**: Bandit, Safety, private key detection
- **File Quality**: Trailing whitespace, line endings, YAML/JSON validation
- **Python Standards**: AST validation, debug statement detection

See [PRE_COMMIT_SETUP.md](PRE_COMMIT_SETUP.md) for detailed documentation.

## System Architecture

```
model_checkpoint/
├── checkpoint/          # Enhanced checkpoint management
│   ├── enhanced_manager.py
│   └── storage/        # Storage backend implementations
├── database/           # Database layer and models
│   ├── enhanced_connection.py
│   └── models.py
├── hooks/              # Hook system for extensibility
│   ├── base_hook.py
│   ├── hook_manager.py
│   ├── quaternion_validation.py
│   ├── grid_monitoring.py
│   └── checkpoint_strategies.py
├── integrity/          # Data integrity verification
├── performance/        # Caching and optimization
└── utils/             # Shared utilities
```

## Hook System

The checkpoint engine features a flexible hook system for extending functionality:

### Available Hook Events
- `BEFORE_CHECKPOINT_SAVE` - Before saving a checkpoint
- `AFTER_CHECKPOINT_SAVE` - After successfully saving a checkpoint  
- `BEFORE_CHECKPOINT_LOAD` - Before loading a checkpoint
- `AFTER_CHECKPOINT_LOAD` - After successfully loading a checkpoint
- `BEFORE_INTEGRITY_CHECK` - Before running integrity verification
- `AFTER_INTEGRITY_CHECK` - After completing integrity verification
- And more...

### Hook Registration Examples

```python
from model_checkpoint.hooks import BaseHook, HookEvent

# Simple function hook
def log_checkpoint_save(context):
    print(f"Saving checkpoint for experiment {context.experiment_id}")
    return True

manager.hook_manager.register_hook(
    "logger",
    log_checkpoint_save,
    [HookEvent.BEFORE_CHECKPOINT_SAVE]
)

# Class-based hook for complex logic
class ValidationHook(BaseHook):
    def execute(self, context):
        # Custom validation logic here
        if context.data.get('loss') > self.max_loss_threshold:
            print("Warning: High loss detected!")
        return True

validation_hook = ValidationHook(priority=10)
manager.hook_manager.register_hook(validation_hook)
```

## Key Features

### Experiment Tracking
- Hierarchical experiment organization with tags and metadata
- Configurable database backends (SQLite, PostgreSQL, MySQL)
- Thread-safe concurrent experiment tracking
- Rich metadata storage for reproducibility

### Enhanced Checkpoint Management
- Best model automatic detection based on multiple criteria
- Configurable retention policies with protected checkpoints
- Integrity verification with SHA256 checksums
- Multiple storage backends (PyTorch native, SafeTensors)
- Compression support for space optimization

### Performance Optimizations
- LRU caching for checkpoint metadata
- Batch processing for database operations
- Connection pooling for improved concurrency
- WAL mode SQLite for better performance

### Extensibility
- Hook system for custom functionality
- Pluggable storage backends
- Event-driven architecture
- Framework-agnostic design

## Usage Examples

### Basic Experiment Tracking
```python
from model_checkpoint import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="resnet_training",
    project_name="image_classification",
    tags=["baseline", "resnet50"],
    config={
        "model": "resnet50",
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": "adam"
    }
)

tracker.log_metric("train_loss", 0.5, step=100)
tracker.log_metric("val_accuracy", 0.85, step=100)
```

### Advanced Checkpoint Management
```python
from model_checkpoint.checkpoint.enhanced_manager import EnhancedCheckpointManager

manager = EnhancedCheckpointManager(
    experiment_tracker=tracker,
    save_best=True,
    save_frequency=5,  # Save every 5 epochs
    max_checkpoints=10,  # Keep max 10 checkpoints
    enable_integrity_checks=True,
    enable_caching=True
)

# Save with automatic best model detection
checkpoint_id = manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    loss=train_loss,
    val_loss=val_loss,
    metrics={"accuracy": val_acc, "f1_score": f1},
    update_best=True
)

# Load best performing checkpoint
best_checkpoint = manager.load_checkpoint(
    checkpoint_type="best_val_loss",
    verify_integrity=True
)
```

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test suites  
python -m pytest tests/test_enhanced_checkpoint.py
python -m pytest tests/test_hooks.py
python -m pytest tests/test_database.py

# Core functionality tests
python -m pytest tests/test_integrity_verification.py
python -m pytest tests/test_experiment_tracker.py
```

## Contributing

This is a generic ML checkpoint management system designed to be framework-agnostic and reusable across different ML projects. When contributing:

1. Keep the system data-agnostic and domain-neutral
2. Maintain backward compatibility
3. Add comprehensive tests for new features
4. Update documentation for API changes
5. Follow the existing code style and patterns

## License

MIT License - see LICENSE file for details.
