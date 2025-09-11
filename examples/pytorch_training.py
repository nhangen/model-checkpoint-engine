#!/usr/bin/env python3
"""
PyTorch Training Integration Example

This example shows how to integrate the Model Checkpoint Engine
with a PyTorch training loop for automatic experiment tracking
and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model_checkpoint import ExperimentTracker, CheckpointManager


# Simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_data(num_samples=1000, input_size=784, num_classes=10):
    """Create dummy dataset for demonstration"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def main():
    """Demonstrate PyTorch integration with checkpoint engine"""
    
    print("🚀 Model Checkpoint Engine - PyTorch Integration Example")
    print("=" * 60)
    
    # Training configuration
    config = {
        'model_type': 'SimpleModel',
        'input_size': 784,
        'hidden_size': 128,
        'num_classes': 10,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    }
    
    # Initialize experiment tracking
    tracker = ExperimentTracker(
        experiment_name="pytorch_simple_model",
        project_name="checkpoint_engine_demo",
        tags=["pytorch", "demo", "simple"],
        config=config
    )
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        tracker=tracker,
        save_best=True,        # Save best performing checkpoint
        save_last=True,        # Save most recent checkpoint  
        save_frequency=2,      # Save every 2 epochs
        max_checkpoints=5      # Keep max 5 checkpoints
    )
    
    # Setup model and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    model = SimpleModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'], 
        num_classes=config['num_classes']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Create datasets
    train_dataset = create_dummy_data(1000)
    val_dataset = create_dummy_data(200)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    
    # Training loop with checkpoint integration
    print("\n🏋️  Starting training...")
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_accuracy = train_correct / len(train_dataset)
        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_dataset)
        
        # Log metrics to tracker
        tracker.log_metrics({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=epoch)
        
        # Save checkpoint
        checkpoint_mgr.save_checkpoint(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            metrics={'val_loss': val_loss, 'val_accuracy': val_accuracy}
        )
        
        print(f"Epoch {epoch+1:2d}/{config['epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.3f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.3f}")
    
    # Complete experiment
    tracker.set_status('completed')
    
    # Generate training report
    print("\n📄 Generating training report...")
    report_path = tracker.generate_report(format_type='html')
    
    # Show checkpoint information
    print("\n💾 Checkpoint Summary:")
    checkpoints = checkpoint_mgr.list_checkpoints()
    for ckpt in checkpoints:
        print(f"   • {ckpt['type']}: Epoch {ckpt['epoch']}, "
              f"Val Loss: {ckpt['metrics']['val_loss']:.4f}")
    
    print(f"\n✅ Training complete!")
    print(f"📊 View training report: {report_path}")
    print(f"💾 Checkpoints saved to: {checkpoint_mgr.checkpoint_dir}")


if __name__ == "__main__":
    main()