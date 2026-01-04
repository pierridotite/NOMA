#!/usr/bin/env python3
"""
Static Model Baseline for MNIST Distribution Shift

This script:
1. Trains a simple MLP on clean MNIST data
2. Evaluates the frozen model on the streaming flux (clean → degraded)
3. Computes rolling accuracy over time
4. Generates visualization with t0 marker and exports metrics to CSV

The purpose is to establish a baseline showing performance degradation
when distribution shift occurs, without any adaptation mechanism.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import sys

try:
    from torchvision import datasets, transforms
    from safetensors.numpy import load_file
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install torch torchvision safetensors matplotlib")
    sys.exit(1)


# ============================================================================
# Model Definition
# ============================================================================

class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MNIST classification.
    
    Architecture:
    - Input: 784 (flattened 28x28)
    - Hidden: 256 → ReLU → 128 → ReLU
    - Output: 10 (digit classes)
    """
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = [256, 128], 
                 num_classes: int = 10, dropout: float = 0.2):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input if needed (handles both [B, 784] and [B, 1, 28, 28])
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


# ============================================================================
# Training
# ============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_clean_mnist(data_root: str = "./mnist_data", batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Load clean MNIST dataset for training.
    
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # No normalization beyond [0,1] to match stream format
    ])
    
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                device: torch.device, epochs: int = 10, lr: float = 1e-3) -> Dict:
    """
    Train the model on clean MNIST.
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    print(f"\n{'='*60}")
    print("Training on Clean MNIST")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total
        
        # Evaluation phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = 100.0 * test_correct / test_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.2f}%, "
              f"Test Acc={test_acc:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Final Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*60}\n")
    
    return history


# ============================================================================
# Streaming Evaluation
# ============================================================================

class StreamLoader:
    """Loads and iterates over the MNIST stream chunks."""
    
    def __init__(self, stream_dir: str):
        self.stream_dir = Path(stream_dir)
        
        # Load manifest
        manifest_path = self.stream_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.n_chunks = self.manifest['stream']['n_chunks']
        self.chunk_files = self.manifest['stream']['chunk_files']
        self.t0 = self.manifest['distribution_shift']['t0']
        self.n_samples = self.manifest['dataset']['n_samples']
        self.image_shape = tuple(self.manifest['format']['image_shape'])
        
        print(f"Stream loaded: {self.n_samples} samples in {self.n_chunks} chunks")
        print(f"Image shape: {self.image_shape}")
        print(f"Distribution shift at t0 = {self.t0}")
    
    def load_chunk(self, chunk_idx: int) -> Dict[str, np.ndarray]:
        """Load a specific chunk."""
        chunk_file = self.chunk_files[chunk_idx]
        chunk_path = self.stream_dir / chunk_file
        return load_file(str(chunk_path))
    
    def iterate_samples(self):
        """
        Iterate over all samples in temporal order.
        
        Yields:
            t: time index
            x: image tensor (numpy)
            y: label (int)
            phase: 'clean' or 'drift'
            intensity: perturbation intensity
        """
        for chunk_idx in range(self.n_chunks):
            chunk = self.load_chunk(chunk_idx)
            n_samples = len(chunk['y'])
            
            for i in range(n_samples):
                yield {
                    't': int(chunk['t'][i]),
                    'x': chunk['x'][i],
                    'y': int(chunk['y'][i]),
                    'phase': 'clean' if chunk['phase'][i] == 0 else 'drift',
                    'intensity': float(chunk['intensity'][i])
                }


def evaluate_streaming(model: nn.Module, stream_dir: str, device: torch.device,
                       window_size: int = 500) -> Dict:
    """
    Evaluate model on streaming data with rolling accuracy.
    
    Args:
        model: Trained model (frozen)
        stream_dir: Path to stream directory
        device: Torch device
        window_size: Window size for rolling accuracy
    
    Returns:
        Dictionary with evaluation results
    """
    model = model.to(device)
    model.eval()
    
    stream = StreamLoader(stream_dir)
    t0 = stream.t0
    
    # Storage for results
    results = {
        't': [],
        'correct': [],
        'predicted': [],
        'actual': [],
        'phase': [],
        'intensity': [],
        'rolling_accuracy': []
    }
    
    # Rolling window buffer
    correct_buffer = []
    
    print(f"\n{'='*60}")
    print("Streaming Evaluation")
    print(f"{'='*60}")
    print(f"Window size for rolling accuracy: {window_size}")
    print(f"Distribution shift at t0 = {t0}")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        for sample in stream.iterate_samples():
            t = sample['t']
            x = torch.from_numpy(sample['x']).unsqueeze(0).to(device)  # [1, 1, 28, 28]
            y = sample['y']
            
            # Forward pass
            output = model(x)
            pred = output.argmax(dim=1).item()
            
            # Check correctness
            is_correct = int(pred == y)
            correct_buffer.append(is_correct)
            
            # Compute rolling accuracy
            if len(correct_buffer) > window_size:
                correct_buffer.pop(0)
            rolling_acc = 100.0 * sum(correct_buffer) / len(correct_buffer)
            
            # Store results
            results['t'].append(t)
            results['correct'].append(is_correct)
            results['predicted'].append(pred)
            results['actual'].append(y)
            results['phase'].append(sample['phase'])
            results['intensity'].append(sample['intensity'])
            results['rolling_accuracy'].append(rolling_acc)
            
            # Progress update
            if t % 5000 == 0:
                phase_str = "CLEAN" if sample['phase'] == 'clean' else f"DRIFT (intensity={sample['intensity']:.2f})"
                print(f"t={t:5d}: Rolling Acc={rolling_acc:.2f}% [{phase_str}]")
    
    # Compute summary statistics
    clean_mask = [p == 'clean' for p in results['phase']]
    drift_mask = [p == 'drift' for p in results['phase']]
    
    clean_correct = [c for c, m in zip(results['correct'], clean_mask) if m]
    drift_correct = [c for c, m in zip(results['correct'], drift_mask) if m]
    
    results['summary'] = {
        't0': t0,
        'n_samples': len(results['t']),
        'window_size': window_size,
        'overall_accuracy': 100.0 * sum(results['correct']) / len(results['correct']),
        'clean_accuracy': 100.0 * sum(clean_correct) / len(clean_correct) if clean_correct else 0,
        'drift_accuracy': 100.0 * sum(drift_correct) / len(drift_correct) if drift_correct else 0,
        'accuracy_drop': None  # Computed below
    }
    
    # Compute accuracy drop (difference between end of clean phase and end of drift phase)
    if clean_correct and drift_correct:
        # Take last 1000 samples of each phase for stable comparison
        clean_end_acc = 100.0 * sum(clean_correct[-1000:]) / len(clean_correct[-1000:])
        drift_end_acc = 100.0 * sum(drift_correct[-1000:]) / len(drift_correct[-1000:])
        results['summary']['accuracy_drop'] = clean_end_acc - drift_end_acc
    
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total samples: {results['summary']['n_samples']}")
    print(f"Overall accuracy: {results['summary']['overall_accuracy']:.2f}%")
    print(f"Clean phase accuracy: {results['summary']['clean_accuracy']:.2f}%")
    print(f"Drift phase accuracy: {results['summary']['drift_accuracy']:.2f}%")
    if results['summary']['accuracy_drop'] is not None:
        print(f"Accuracy drop: {results['summary']['accuracy_drop']:.2f}%")
    print(f"{'='*60}\n")
    
    return results


# ============================================================================
# Visualization & Export
# ============================================================================

def plot_results(results: Dict, output_path: str, title: str = "Static Model Baseline"):
    """
    Generate visualization of streaming evaluation results.
    
    Creates a plot showing:
    - Rolling accuracy over time
    - Vertical line at t0 (distribution shift)
    - Phase regions (clean vs drift)
    """
    t = results['t']
    rolling_acc = results['rolling_accuracy']
    intensity = results['intensity']
    t0 = results['summary']['t0']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Main plot: Rolling accuracy
    ax1.plot(t, rolling_acc, 'b-', linewidth=1.0, alpha=0.8, label='Rolling Accuracy')
    
    # Add t0 marker
    ax1.axvline(x=t0, color='red', linestyle='--', linewidth=2, label=f't0 = {t0}')
    
    # Shade regions
    ax1.axvspan(0, t0, alpha=0.1, color='green', label='Clean Phase')
    ax1.axvspan(t0, max(t), alpha=0.1, color='red', label='Drift Phase')
    
    # Labels and formatting
    ax1.set_ylabel('Rolling Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, max(t))
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{title}\n'
                  f'Window={results["summary"]["window_size"]}, '
                  f'Clean Acc={results["summary"]["clean_accuracy"]:.1f}%, '
                  f'Drift Acc={results["summary"]["drift_accuracy"]:.1f}%',
                  fontsize=14)
    
    # Add horizontal line at key accuracy levels
    ax1.axhline(y=results['summary']['clean_accuracy'], color='green', 
                linestyle=':', alpha=0.5, linewidth=1)
    ax1.axhline(y=results['summary']['drift_accuracy'], color='red', 
                linestyle=':', alpha=0.5, linewidth=1)
    
    # Secondary plot: Perturbation intensity
    ax2.fill_between(t, intensity, alpha=0.5, color='orange', label='Perturbation Intensity')
    ax2.axvline(x=t0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def export_metrics(results: Dict, output_path: str):
    """Export metrics to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'correct', 'predicted', 'actual', 'phase', 
                        'intensity', 'rolling_accuracy'])
        
        for i in range(len(results['t'])):
            writer.writerow([
                results['t'][i],
                results['correct'][i],
                results['predicted'][i],
                results['actual'][i],
                results['phase'][i],
                results['intensity'][i],
                results['rolling_accuracy'][i]
            ])
    
    print(f"Metrics exported to: {output_path}")


def export_summary(results: Dict, training_history: Dict, output_path: str):
    """Export summary statistics to JSON."""
    summary = {
        'experiment': 'static_model_baseline',
        'timestamp': datetime.now().isoformat(),
        'training': {
            'epochs': len(training_history['train_loss']),
            'final_train_acc': training_history['train_acc'][-1],
            'final_test_acc': training_history['test_acc'][-1]
        },
        'evaluation': results['summary']
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary exported to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Static Model Baseline for MNIST Distribution Shift',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--stream-dir', type=str, 
                        default='./sequencer/mnist_stream',
                        help='Directory containing the MNIST stream')
    parser.add_argument('--data-root', type=str,
                        default='./sequencer/mnist_data',
                        help='Root directory for MNIST data (training)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model arguments
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 128],
                        help='Hidden layer sizes for MLP')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Evaluation arguments
    parser.add_argument('--window-size', type=int, default=500,
                        help='Window size for rolling accuracy')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./baseline/output',
                        help='Directory for output files')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print("# Static Model Baseline for MNIST Distribution Shift")
    print(f"{'#'*60}")
    print(f"\nConfiguration:")
    print(f"  Stream directory: {args.stream_dir}")
    print(f"  Training data: {args.data_root}")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"  Output directory: {output_dir}")
    
    # Step 1: Load clean MNIST for training
    print("\n[Step 1] Loading clean MNIST for training...")
    train_loader, test_loader = load_clean_mnist(args.data_root, args.batch_size)
    
    # Step 2: Create and train model
    print("\n[Step 2] Training model on clean MNIST...")
    model = SimpleMLP(
        input_size=784,
        hidden_sizes=args.hidden_sizes,
        num_classes=10,
        dropout=args.dropout
    )
    print(f"Model architecture: {model}")
    
    training_history = train_model(
        model, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr
    )
    
    # Optional: Save model checkpoint
    if args.save_model:
        model_path = output_dir / 'static_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': training_history,
            'args': vars(args)
        }, model_path)
        print(f"Model saved to: {model_path}")
    
    # Step 3: Evaluate on streaming data
    print("\n[Step 3] Evaluating on streaming data...")
    results = evaluate_streaming(model, args.stream_dir, device, args.window_size)
    
    # Step 4: Generate outputs
    print("\n[Step 4] Generating outputs...")
    
    # Plot
    plot_path = output_dir / 'static_baseline_accuracy.png'
    plot_results(results, str(plot_path))
    
    # Metrics CSV
    csv_path = output_dir / 'static_baseline_metrics.csv'
    export_metrics(results, str(csv_path))
    
    # Summary JSON
    summary_path = output_dir / 'static_baseline_summary.json'
    export_summary(results, training_history, str(summary_path))
    
    print(f"\n{'#'*60}")
    print("# Baseline Complete!")
    print(f"{'#'*60}")
    print(f"\nOutputs generated in: {output_dir}")
    print(f"  - {plot_path.name}: Accuracy over time visualization")
    print(f"  - {csv_path.name}: Detailed metrics (t, correct, rolling_accuracy, ...)")
    print(f"  - {summary_path.name}: Summary statistics")
    print(f"\nKey Results:")
    print(f"  Clean phase accuracy:  {results['summary']['clean_accuracy']:.2f}%")
    print(f"  Drift phase accuracy:  {results['summary']['drift_accuracy']:.2f}%")
    if results['summary']['accuracy_drop'] is not None:
        print(f"  Accuracy drop:         {results['summary']['accuracy_drop']:.2f}%")


if __name__ == '__main__':
    main()
