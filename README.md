# ML Checkpoint & Experiment Tracking System

A comprehensive experiment tracking and checkpoint management system for machine learning projects.

## Features

- **Checkpoint Management**: Intelligent model saving with metadata
- **Experiment Tracking**: SQL-based experiment logging with reports
- **Performance Analytics**: Automated training reports and visualizations
- **Multi-framework Support**: PyTorch, TensorFlow, JAX compatibility
- **Database Integration**: SQLite/PostgreSQL for experiment data
- **Report Generation**: Automated HTML/PDF training reports

## Quick Start

```python
from ml_checkpoint import ExperimentTracker, CheckpointManager

# Initialize tracking
tracker = ExperimentTracker(experiment_name="aircraft_pose_vit")
checkpoint_mgr = CheckpointManager(tracker)

# During training
for epoch in range(epochs):
    # ... training loop ...
    
    # Log metrics
    tracker.log_metrics({
        'train_loss': loss,
        'val_accuracy': accuracy,
        'rotation_error': rotation_error
    })
    
    # Save checkpoint
    if epoch % 5 == 0:
        checkpoint_mgr.save_checkpoint(model, optimizer, epoch)

# Generate report
tracker.generate_report(format='html')
```

## Installation

```bash
pip install ml-checkpoint-system
```

## Repository Structure

See `docs/architecture.md` for detailed system design.