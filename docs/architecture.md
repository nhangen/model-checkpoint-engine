# ML Checkpoint & Experiment Tracking System Architecture

## Repository Structure

```
ml-checkpoint-system/
├── README.md                 # Main documentation
├── LICENSE                   # MIT license
├── setup.py                  # Package installation
├── requirements.txt          # Core dependencies
├── requirements-dev.txt      # Development dependencies
├── examples/                 # Usage examples
│   ├── pytorch_training.py   # PyTorch integration example
│   ├── tensorflow_training.py # TensorFlow example
│   └── custom_metrics.py     # Custom metrics example
├── ml_checkpoint/            # Main package
│   ├── __init__.py
│   ├── core/                 # Core tracking system
│   │   ├── __init__.py
│   │   ├── experiment.py     # Experiment tracking
│   │   ├── checkpoint.py     # Checkpoint management
│   │   ├── metrics.py        # Metrics collection
│   │   └── database.py       # Database operations
│   ├── storage/              # Storage backends
│   │   ├── __init__.py
│   │   ├── local.py          # Local filesystem
│   │   ├── s3.py            # AWS S3 backend
│   │   └── gcs.py           # Google Cloud Storage
│   ├── frameworks/           # ML framework integrations
│   │   ├── __init__.py
│   │   ├── pytorch.py        # PyTorch helpers
│   │   ├── tensorflow.py     # TensorFlow helpers
│   │   └── jax.py           # JAX helpers
│   ├── reporting/            # Report generation
│   │   ├── __init__.py
│   │   ├── html.py          # HTML reports
│   │   ├── pdf.py           # PDF reports
│   │   ├── templates/        # Report templates
│   │   └── visualizations.py # Plotting utilities
│   ├── database/             # Database schemas
│   │   ├── __init__.py
│   │   ├── models.py         # SQLAlchemy models
│   │   ├── migrations/       # Database migrations
│   │   └── queries.py        # Common queries
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── logging.py        # Enhanced logging
│       └── validation.py     # Data validation
├── tests/                    # Unit tests
│   ├── test_experiment.py
│   ├── test_checkpoint.py
│   ├── test_reporting.py
│   └── test_database.py
├── docs/                     # Documentation
│   ├── quickstart.md
│   ├── api_reference.md
│   ├── integrations.md
│   └── deployment.md
├── scripts/                  # Utility scripts
│   ├── migrate_db.py         # Database migration
│   ├── cleanup_old.py        # Cleanup old experiments
│   └── export_data.py        # Data export utilities
└── docker/                   # Docker deployment
    ├── Dockerfile
    ├── docker-compose.yml
    └── init.sql
```

## Core Components

### 1. Experiment Tracker
```python
class ExperimentTracker:
    def __init__(self,
                 experiment_name: str,
                 project_name: str = None,
                 tags: List[str] = None,
                 config: Dict = None,
                 storage_backend: str = 'local'):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log training metrics"""
        pass

    def log_hyperparameters(self, params: Dict):
        """Log hyperparameters"""
        pass

    def log_artifacts(self, artifacts: Dict[str, Any]):
        """Log model artifacts, plots, etc."""
        pass

    def set_status(self, status: str):
        """Set experiment status (running, completed, failed)"""
        pass
```

### 2. Checkpoint Manager
```python
class CheckpointManager:
    def __init__(self,
                 tracker: ExperimentTracker,
                 save_best: bool = True,
                 save_last: bool = True,
                 save_frequency: int = 5,
                 max_checkpoints: int = 10):
        pass

    def save_checkpoint(self,
                       model,
                       optimizer=None,
                       epoch: int = None,
                       metrics: Dict = None,
                       metadata: Dict = None):
        """Save model checkpoint with metadata"""
        pass

    def load_checkpoint(self, checkpoint_id: str = 'best'):
        """Load checkpoint by ID or 'best'/'last'"""
        pass

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List available checkpoints"""
        pass

    def cleanup_old_checkpoints(self):
        """Remove old checkpoints based on retention policy"""
        pass
```

### 3. Database Schema
```sql
-- Experiments table
CREATE TABLE experiments (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    project_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'running',
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,
    created_by VARCHAR(255),
    tags TEXT[],
    config JSONB,
    metadata JSONB
);

-- Metrics table
CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(id),
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    step INTEGER,
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Checkpoints table
CREATE TABLE checkpoints (
    id UUID PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(id),
    epoch INTEGER,
    checkpoint_type VARCHAR(50), -- 'best', 'last', 'manual'
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    sha256_hash VARCHAR(64),
    metrics JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Artifacts table
CREATE TABLE artifacts (
    id UUID PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(id),
    artifact_name VARCHAR(255) NOT NULL,
    artifact_type VARCHAR(100), -- 'plot', 'model', 'data', 'log'
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 4. Report Generation
```python
class ReportGenerator:
    def __init__(self, experiment_tracker: ExperimentTracker):
        pass

    def generate_training_report(self,
                               format: str = 'html',
                               include_plots: bool = True,
                               include_metrics: bool = True) -> str:
        """Generate comprehensive training report"""
        pass

    def generate_comparison_report(self,
                                 experiment_ids: List[str],
                                 format: str = 'html') -> str:
        """Compare multiple experiments"""
        pass

    def generate_dashboard(self, project_name: str = None) -> str:
        """Generate live dashboard"""
        pass
```

## Usage Examples

### 1. Basic Training Loop Integration
```python
from ml_checkpoint import ExperimentTracker, CheckpointManager

# Initialize
tracker = ExperimentTracker(
    experiment_name="aircraft_pose_vit_v2",
    project_name="aircraft_pose_estimation",
    tags=["vit", "pose", "aircraft"],
    config={
        "model": "vit-base",
        "batch_size": 16,
        "learning_rate": 5e-5,
        "epochs": 50
    }
)

checkpoint_mgr = CheckpointManager(
    tracker=tracker,
    save_best=True,
    save_frequency=5,
    max_checkpoints=10
)

# Training loop
for epoch in range(50):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_metrics = validate(model, val_loader)

    # Log metrics
    tracker.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'rotation_error': val_metrics['rotation_error'],
        'translation_error': val_metrics['translation_error'],
        'learning_rate': optimizer.param_groups[0]['lr']
    }, step=epoch)

    # Save checkpoint
    checkpoint_mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=val_metrics
    )

    # Log plots
    if epoch % 10 == 0:
        plot_path = create_loss_plot(tracker.get_metrics())
        tracker.log_artifacts({'loss_plot': plot_path})

# Generate final report
tracker.set_status('completed')
report_path = tracker.generate_report(format='html')
print(f"Training report: {report_path}")
```

### 2. Advanced Features
```python
# Resume from checkpoint
tracker = ExperimentTracker.resume(experiment_id="uuid-here")
checkpoint_mgr = CheckpointManager(tracker)
model, optimizer, start_epoch = checkpoint_mgr.load_checkpoint('best')

# Compare experiments
from ml_checkpoint.reporting import ComparisonReport
comparison = ComparisonReport([exp1_id, exp2_id, exp3_id])
comparison.generate_html_report('model_comparison.html')

# Query experiments
from ml_checkpoint.database import ExperimentQuery
query = ExperimentQuery()
best_experiments = query.get_top_experiments(
    project='aircraft_pose_estimation',
    metric='rotation_error',
    limit=5
)
```

### 3. Custom Metrics and Callbacks
```python
# Custom metric tracking
class RotationErrorMetric:
    def __init__(self, tracker):
        self.tracker = tracker
        self.errors = []

    def update(self, pred_rotation, true_rotation):
        error = calculate_rotation_error(pred_rotation, true_rotation)
        self.errors.append(error)

    def compute_and_log(self, step):
        avg_error = np.mean(self.errors)
        self.tracker.log_metrics({'rotation_error': avg_error}, step)
        self.errors.clear()

# PyTorch callback integration
from ml_checkpoint.frameworks.pytorch import PyTorchCallback
callback = PyTorchCallback(tracker, checkpoint_mgr)
trainer.add_callback(callback)
```

## Database Features

### Performance Analytics
```sql
-- Get experiment performance trends
SELECT
    e.name,
    e.start_time,
    MIN(m.metric_value) as best_rotation_error,
    AVG(m.metric_value) as avg_rotation_error
FROM experiments e
JOIN metrics m ON e.id = m.experiment_id
WHERE m.metric_name = 'rotation_error'
GROUP BY e.id, e.name, e.start_time
ORDER BY best_rotation_error;

-- Find best hyperparameters
SELECT
    config->>'learning_rate' as lr,
    config->>'batch_size' as batch_size,
    MIN(final_metrics.rotation_error) as best_error
FROM experiments e
JOIN (
    SELECT experiment_id,
           MIN(metric_value) as rotation_error
    FROM metrics
    WHERE metric_name = 'rotation_error'
    GROUP BY experiment_id
) final_metrics ON e.id = final_metrics.experiment_id
GROUP BY config->>'learning_rate', config->>'batch_size'
ORDER BY best_error;
```

## Deployment Options

### 1. Local Development
```bash
pip install ml-checkpoint-system
```

### 2. Team/Production Setup
```yaml
# docker-compose.yml
version: '3.8'
services:
  ml-checkpoint-db:
    image: postgres:13
    environment:
      POSTGRES_DB: ml_experiments
      POSTGRES_USER: ml_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/init.sql:/docker-entrypoint-initdb.d/init.sql

  ml-checkpoint-web:
    image: ml-checkpoint-system:latest
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://ml_user:secure_password@ml-checkpoint-db:5432/ml_experiments
    depends_on:
      - ml-checkpoint-db
```

## Configuration Management
```python
# ml_checkpoint/config.py
class Config:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///experiments.db')
        self.storage_backend = os.getenv('STORAGE_BACKEND', 'local')
        self.checkpoint_dir = os.getenv('CHECKPOINT_DIR', './checkpoints')
        self.max_checkpoint_retention = int(os.getenv('MAX_CHECKPOINTS', '10'))
        self.report_template_dir = os.getenv('REPORT_TEMPLATES', './templates')
```

This architecture provides a complete, production-ready system for ML experiment tracking with database integration, comprehensive reporting, and multi-framework support.
