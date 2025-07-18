# 🚀 Complete Training Guide: Custom Manim LLM

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [System Architecture](#system-architecture)
4. [Installation Guide](#installation-guide)
5. [Training Process](#training-process)
6. [Configuration Options](#configuration-options)
7. [Monitoring & Evaluation](#monitoring--evaluation)
8. [Advanced Training Techniques](#advanced-training-techniques)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)
11. [Usage Examples](#usage-examples)
12. [FAQ](#faq)

---

## Overview

This guide walks you through training your own custom Large Language Model (LLM) specifically designed for generating Manim animation scripts. Unlike traditional approaches that rely on external APIs, this system creates a completely local, specialized model that understands Manim syntax and mathematical animation patterns.

### Key Features
- 🧠 **Custom Transformer Architecture**: Built from scratch for Manim
- 📊 **Synthetic Data Generation**: Creates 3000+ training samples automatically
- 🏠 **Fully Local**: No external API dependencies
- 🎯 **Specialized Tokenizer**: Understands Python + Manim syntax
- ✅ **Built-in Validation**: Automatic script checking and fixing
- 📈 **Real-time Monitoring**: Training progress with visual feedback

### Training Specifications
- **Architecture**: Decoder-only transformer
- **Default Model Size**: 25M parameters (configurable)
- **Training Data**: 3000 synthetic Manim script pairs
- **Vocabulary**: 8000 Manim-specific tokens
- **Training Time**: 30-60 minutes on CPU, 10-20 minutes on GPU

---

## Prerequisites

### System Requirements

#### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Storage**: 2GB free space
- **CPU**: Multi-core processor (Intel i5 or AMD Ryzen 5)
- **OS**: Windows 10, macOS, or Linux

#### Recommended Requirements
- **Python**: 3.9 or 3.10
- **RAM**: 16GB or higher
- **Storage**: 5GB free space
- **CPU**: Intel i7/AMD Ryzen 7 or better
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Software Dependencies

#### Core Dependencies
```bash
torch>=2.0.0          # Neural network framework
numpy>=1.24.0         # Numerical computing
tokenizers>=0.15.0    # Text processing
matplotlib>=3.7.0     # Plotting and visualization
scikit-learn>=1.3.0   # Machine learning utilities
tqdm>=4.66.0          # Progress bars
```

#### Optional Dependencies
```bash
manim>=0.18.0         # For testing generated scripts
tensorboard>=2.15.0   # Advanced training monitoring
wandb>=0.16.0         # Experiment tracking
accelerate>=0.25.0    # GPU acceleration
```

---

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CUSTOM MANIM LLM SYSTEM                 │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Data Generator │  │    Tokenizer    │  │     Model       │
│                 │  │                 │  │                 │
│ • Synthetic     │  │ • Manim Vocab   │  │ • Transformer   │
│   Samples       │  │ • Python Code   │  │ • Multi-Head    │
│ • Request-      │  │ • Math Symbols  │  │   Attention     │
│   Response      │  │ • 8K Tokens     │  │ • 25M Params    │
│ • Validation    │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                                │
                                ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    Trainer      │  │   Validator     │  │ Inference Engine│
│                 │  │                 │  │                 │
│ • Training Loop │  │ • Syntax Check  │  │ • Text Generation│
│ • Checkpoints   │  │ • Code Fixing   │  │ • Quality Control│
│ • Monitoring    │  │ • Best Practices│  │ • Script Output │
│ • Evaluation    │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Data Flow

```
User Request → Tokenizer → Model → Decoder → Validator → Final Script
     │              │         │        │         │           │
     │              │         │        │         │           │
"Create a    →  [1,45,23,8] → Neural → Raw   → Syntax  → Python
blue circle"                   Network   Text    Check      Code
```

---

## Installation Guide

### Step 1: Environment Setup

#### Option A: Using pip (Recommended)
```bash
# Navigate to your project directory
cd /path/to/manim_agent

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all dependencies
pip install -r requirements.txt
```

#### Option B: Using conda
```bash
# Create new conda environment
conda create -n manim_llm python=3.9
conda activate manim_llm

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

#### Option C: GPU Support (NVIDIA only)
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Step 2: System Verification

```bash
# Run comprehensive system test
python test_system.py
```

**Expected Output:**
```
🧪 SYSTEM TESTING
Testing all components before training...

✅ Knowledge base works
✅ Single generation works
✅ Batch generation works
✅ Tokenizer works
✅ Model creation works
✅ Validation works

🎉 ALL TESTS PASSED!
Your system is ready for training!
```

---

## Training Process

### Overview of Training Phases

The training process consists of five distinct phases:

1. **Data Generation**: Creates synthetic training samples
2. **Tokenization**: Builds specialized vocabulary
3. **Model Creation**: Initializes transformer architecture
4. **Training Loop**: Learns from data over multiple epochs
5. **Evaluation**: Tests model performance

### Phase 1: Data Generation

#### Automatic Data Generation
```bash
# Generate default dataset (3000 samples)
python train_model.py --data-size 3000
```

#### What Happens:
- Creates request-response pairs: `"Create a blue circle"` → Python script
- Generates diverse patterns: shapes, animations, text, math
- Splits into training (80%) and validation (20%)
- Saves as JSON files for reuse

#### Sample Generated Data:
```json
{
  "id": 1,
  "request": "Create a blue circle that rotates",
  "script": "from manim import *\n\nclass CircleScene(Scene):\n    def construct(self):\n        circle = Circle(radius=1)\n        circle.set_color(BLUE)\n        self.play(Create(circle))\n        self.play(Rotate(circle, angle=PI/2))\n        self.wait(1)",
  "length": 198
}
```

#### Data Categories:
- **Basic Shapes**: Circles, squares, triangles, rectangles
- **Animations**: Create, fade, transform, rotate, scale
- **Text & Math**: LaTeX formulas, equations, labels
- **Complex Scenes**: Multiple objects, coordinated animations
- **Colors & Positioning**: RGB colors, spatial arrangements

### Phase 2: Tokenization

#### Tokenizer Creation
```bash
# Build specialized tokenizer
python -c "from tokenizer import build_tokenizer_from_data; import json; data = json.load(open('train_data.json')); tokenizer = build_tokenizer_from_data(data, 8000); tokenizer.save('tokenizer.pkl')"
```

#### Tokenization Process:
1. **Text Analysis**: Processes all training text
2. **Vocabulary Building**: Identifies most common tokens
3. **Special Tokens**: Adds `<BOS>`, `<EOS>`, `<REQ>`, `<PAD>`
4. **Code Patterns**: Handles Python syntax, function calls
5. **Math Symbols**: Supports LaTeX mathematical notation

#### Tokenizer Statistics:
```
Vocabulary Size: 8000 tokens
Special Tokens: 6 (<BOS>, <EOS>, <REQ>, <PAD>, <UNK>, <SCR>)
Python Keywords: 20 (def, class, if, for, etc.)
Manim Keywords: 50+ (Scene, Circle, Create, etc.)
Math Symbols: 100+ (LaTeX expressions)
```

### Phase 3: Model Architecture

#### Transformer Specifications
```python
# Default model configuration
ManimLLMConfig(
    vocab_size=8000,       # Vocabulary size
    d_model=512,          # Model dimension
    n_heads=8,            # Attention heads
    n_layers=6,           # Transformer layers
    d_ff=2048,            # Feed-forward dimension
    max_len=1024          # Maximum sequence length
)
```

#### Architecture Details:
- **Decoder-Only**: Autoregressive generation
- **Multi-Head Attention**: 8 attention heads
- **Positional Encoding**: Sinusoidal embeddings
- **Layer Normalization**: Pre-norm configuration
- **Dropout**: 0.1 for regularization

#### Model Sizes:
| Size | Parameters | d_model | n_layers | Training Time |
|------|------------|---------|----------|---------------|
| Small | 5M | 256 | 4 | 15 min |
| Medium | 25M | 512 | 6 | 45 min |
| Large | 100M | 1024 | 12 | 3 hours |

### Phase 4: Training Loop

#### Training Command
```bash
# Start training with default settings
python train_model.py
```

#### Training Configuration:
```python
training_config = {
    'learning_rate': 1e-4,        # Learning rate
    'weight_decay': 0.01,         # L2 regularization
    'batch_size': 4,              # Batch size
    'epochs': 20,                 # Training epochs
    'max_length': 512,            # Sequence length
    'optimizer': 'AdamW',         # Optimizer
    'scheduler': 'CosineAnnealingLR'  # Learning rate scheduler
}
```

#### Training Process:
1. **Data Loading**: Loads training and validation data
2. **Model Initialization**: Creates transformer model
3. **Training Loop**: Iterates through epochs
4. **Validation**: Evaluates on held-out data
5. **Checkpointing**: Saves best models
6. **Monitoring**: Tracks loss, perplexity, learning rate

#### Sample Training Output:
```
Starting training for 20 epochs...
Device: cuda
Model parameters: 25,641,000

Epoch 1/20
--------------------------------------------------
Training: 100%|██████████| 600/600 [05:23<00:00, 1.85it/s]
loss: 4.2341, lr: 0.000100
Validating: 100%|██████████| 150/150 [01:12<00:00, 2.08it/s]
loss: 3.8745

Train Loss: 4.23, Train Perplexity: 68.4
Val Loss: 3.87, Val Perplexity: 48.2
Learning Rate: 0.000100
New best model saved: best_model_epoch_1.pth

Epoch 2/20
--------------------------------------------------
Training: 100%|██████████| 600/600 [05:18<00:00, 1.89it/s]
loss: 3.9823, lr: 0.000099
Validating: 100%|██████████| 150/150 [01:10<00:00, 2.14it/s]
loss: 3.5632

Train Loss: 3.98, Train Perplexity: 53.7
Val Loss: 3.56, Val Perplexity: 35.2
Learning Rate: 0.000099
New best model saved: best_model_epoch_2.pth

...

Epoch 20/20
--------------------------------------------------
Training: 100%|██████████| 600/600 [05:02<00:00, 1.98it/s]
loss: 1.2456, lr: 0.000001
Validating: 100%|██████████| 150/150 [01:08<00:00, 2.20it/s]
loss: 1.3421

Train Loss: 1.25, Train Perplexity: 3.5
Val Loss: 1.34, Val Perplexity: 3.8
Learning Rate: 0.000001

Training completed in 1847.32 seconds
Best validation loss: 1.234
```

### Phase 5: Evaluation

#### Automatic Evaluation
```bash
# Evaluate trained model
python evaluate_model.py --model-path best_model_epoch_15.pth
```

#### Evaluation Metrics:
- **Validation Loss**: How well the model predicts
- **Perplexity**: Model confidence (lower is better)
- **Script Quality**: Syntax correctness, completeness
- **Category Performance**: Performance across different request types

#### Sample Evaluation Results:
```
MANIM LLM EVALUATION REPORT
============================================================

Overall Statistics:
  Total Tests: 30
  Average Score: 87.3
  Total Score: 2619.0

BASIC SHAPES
----------------------------------------
Average Score: 92.4
Tests: 5

Test 1: Create a blue circle
Score: 95.0
Generated Script:
from manim import *

class CircleScene(Scene):
    def construct(self):
        circle = Circle(radius=1)
        circle.set_color(BLUE)
        self.play(Create(circle))
        self.wait(1)

ANIMATIONS
----------------------------------------
Average Score: 88.6
Tests: 5

TRANSFORMATIONS
----------------------------------------
Average Score: 85.2
Tests: 5

TEXT AND MATH
----------------------------------------
Average Score: 89.8
Tests: 5

COMPLEX SCENES
----------------------------------------
Average Score: 82.1
Tests: 5

COLORS AND POSITIONING
----------------------------------------
Average Score: 86.5
Tests: 5
```

---

## Configuration Options

### Training Parameters

#### Data Configuration
```bash
# Data generation options
--data-size 3000        # Number of training samples
--val-split 0.2         # Validation split ratio
--max-length 512        # Maximum sequence length
--seed 42               # Random seed for reproducibility
```

#### Model Architecture
```bash
# Model size options
--d-model 512           # Model dimension (256, 512, 1024)
--n-layers 6            # Number of transformer layers
--n-heads 8             # Number of attention heads
--d-ff 2048             # Feed-forward dimension
--vocab-size 8000       # Vocabulary size
--max-len 1024          # Maximum sequence length
```

#### Training Hyperparameters
```bash
# Training options
--epochs 20             # Number of training epochs
--batch-size 4          # Batch size
--learning-rate 1e-4    # Learning rate
--weight-decay 0.01     # L2 regularization
--dropout 0.1           # Dropout rate
--grad-clip 1.0         # Gradient clipping
```

#### Hardware Configuration
```bash
# Hardware options
--device auto           # auto, cpu, cuda
--num-workers 2         # DataLoader workers
--pin-memory true       # Pin memory for GPU
```

### Predefined Configurations

#### Quick Test Configuration
```bash
python train_model.py \
    --data-size 1000 \
    --epochs 5 \
    --d-model 256 \
    --n-layers 4 \
    --batch-size 8
```

#### Balanced Configuration
```bash
python train_model.py \
    --data-size 3000 \
    --epochs 20 \
    --d-model 512 \
    --n-layers 6 \
    --batch-size 4
```

#### High-Quality Configuration
```bash
python train_model.py \
    --data-size 5000 \
    --epochs 30 \
    --d-model 768 \
    --n-layers 8 \
    --batch-size 2
```

#### Production Configuration
```bash
python train_model.py \
    --data-size 10000 \
    --epochs 50 \
    --d-model 1024 \
    --n-layers 12 \
    --batch-size 1 \
    --device cuda
```

---

## Monitoring & Evaluation

### Real-time Monitoring

#### Training Metrics
- **Loss**: Cross-entropy loss (lower is better)
- **Perplexity**: exp(loss) - model confidence
- **Learning Rate**: Current learning rate
- **Gradient Norm**: Gradient magnitude
- **Memory Usage**: GPU/CPU memory consumption

#### Validation Metrics
- **Validation Loss**: Performance on unseen data
- **Overfitting Detection**: Train vs validation loss gap
- **Early Stopping**: Automatic training termination
- **Best Model Tracking**: Saves best performing checkpoint

### Training Visualization

#### Training Curves
The system automatically generates `training_history.png`:
- Loss curves (train vs validation)
- Perplexity over time
- Learning rate schedule
- Loss difference (overfitting indicator)

#### Example Training Plot:
```
Training Loss: 4.2 → 3.8 → 3.2 → 2.1 → 1.4 → 1.2
Validation Loss: 4.5 → 4.1 → 3.6 → 2.8 → 2.1 → 1.8
```

### Checkpoint Management

#### Automatic Checkpointing
- **Best Model**: `best_model_epoch_X.pth`
- **Regular Checkpoints**: `checkpoint_epoch_X.pth` (every 5 epochs)
- **Final Model**: `final_manim_model.pth`

#### Checkpoint Contents:
```python
checkpoint = {
    'epoch': 15,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_loss': 1.234,
    'train_losses': [4.2, 3.8, 3.2, ...],
    'val_losses': [4.5, 4.1, 3.6, ...],
    'config': training_config
}
```

### Performance Evaluation

#### Comprehensive Evaluation
```bash
# Run full evaluation suite
python evaluate_model.py --model-path best_model_epoch_15.pth --output-dir evaluation_results
```

#### Evaluation Categories:
1. **Basic Shapes**: Simple geometric objects
2. **Animations**: Movement and transformations
3. **Text & Math**: LaTeX formulas and text
4. **Complex Scenes**: Multi-object animations
5. **Colors & Positioning**: Visual styling

#### Scoring System:
- **Syntax Correctness**: 50 points
- **Completeness**: 30 points
- **Best Practices**: 20 points
- **Bonus**: Animation quality, style

#### Interactive Evaluation
```bash
# Test model interactively
python evaluate_model.py --interactive
```

---

## Advanced Training Techniques

### Curriculum Learning

#### Stage-based Training
```python
# Stage 1: Basic shapes (epochs 1-10)
python train_model.py --data-size 2000 --epochs 10 --focus basic_shapes

# Stage 2: Animations (epochs 11-20)
python train_model.py --resume best_model_epoch_10.pth --epochs 10 --focus animations

# Stage 3: Complex scenes (epochs 21-30)
python train_model.py --resume best_model_epoch_20.pth --epochs 10 --focus complex
```

#### Progressive Difficulty
```python
# Gradually increase complexity
difficulty_levels = [
    "simple_shapes",      # Epoch 1-5
    "basic_animations",   # Epoch 6-10
    "text_and_math",     # Epoch 11-15
    "transformations",   # Epoch 16-20
    "complex_scenes"     # Epoch 21-25
]
```

### Data Augmentation

#### Synthetic Data Expansion
```python
# Generate domain-specific data
def generate_physics_data():
    """Generate physics-focused training data"""
    requests = [
        "Create a pendulum that swings",
        "Show projectile motion",
        "Animate wave interference",
        "Display Newton's laws"
    ]
    # Generate corresponding scripts...

def generate_math_data():
    """Generate mathematics-focused training data"""
    requests = [
        "Show the quadratic formula",
        "Animate a sine wave",
        "Display matrix multiplication",
        "Create a coordinate system"
    ]
    # Generate corresponding scripts...
```

#### Custom Example Integration
```python
# Add your own high-quality examples
CUSTOM_EXAMPLES = {
    "advanced_pendulum": """
from manim import *
import numpy as np

class Pendulum(Scene):
    def construct(self):
        # Create pendulum components
        pivot = Dot(UP * 2, color=BLACK)
        bob = Circle(radius=0.2, color=RED, fill_opacity=1)
        
        # Initial position
        angle = PI/6
        bob.move_to(pivot.get_center() + 2 * np.array([np.sin(angle), -np.cos(angle), 0]))
        
        # Create string
        string = Line(pivot.get_center(), bob.get_center(), color=GRAY)
        
        # Group pendulum
        pendulum = VGroup(pivot, string, bob)
        
        # Show pendulum
        self.play(Create(pendulum))
        
        # Animate swinging
        for i in range(3):
            self.play(
                Rotate(pendulum, angle=PI/3, about_point=pivot.get_center()),
                run_time=1
            )
            self.play(
                Rotate(pendulum, angle=-PI/3, about_point=pivot.get_center()),
                run_time=1
            )
        
        self.wait(1)
    """,
    
    "matrix_visualization": """
from manim import *

class MatrixMultiplication(Scene):
    def construct(self):
        # Create matrices
        matrix_a = Matrix([["a", "b"], ["c", "d"]])
        matrix_b = Matrix([["e", "f"], ["g", "h"]])
        result = Matrix([["ae+bg", "af+bh"], ["ce+dg", "cf+dh"]])
        
        # Position matrices
        matrix_a.shift(LEFT * 3)
        matrix_b.shift(LEFT * 1)
        result.shift(RIGHT * 2)
        
        # Show multiplication
        self.play(Write(matrix_a))
        self.play(Write(matrix_b))
        self.play(Transform(VGroup(matrix_a, matrix_b), result))
        
        self.wait(2)
    """
}
```

### Hyperparameter Optimization

#### Grid Search
```python
# Hyperparameter combinations
hyperparams = {
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'batch_size': [2, 4, 8],
    'd_model': [256, 512, 768],
    'n_layers': [4, 6, 8]
}

# Run grid search
best_params = grid_search(hyperparams, train_data, val_data)
```

#### Learning Rate Scheduling
```python
# Different scheduling strategies
schedulers = {
    'cosine': CosineAnnealingLR,
    'exponential': ExponentialLR,
    'reduce_plateau': ReduceLROnPlateau,
    'warmup_cosine': WarmupCosineScheduler
}
```

### Transfer Learning

#### Pre-trained Initialization
```python
# Initialize from general language model
def initialize_from_pretrained(model, pretrained_path):
    """Initialize model weights from pre-trained model"""
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    
    # Filter compatible layers
    compatible_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
```

#### Fine-tuning Strategies
```python
# Freeze early layers, train later layers
def freeze_early_layers(model, freeze_layers=4):
    """Freeze first N transformer layers"""
    for i, layer in enumerate(model.transformer_blocks):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Installation Problems

**Problem**: PyTorch installation fails
```bash
ERROR: Could not find a version that satisfies the requirement torch
```

**Solution**:
```bash
# Try different PyTorch versions
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0

# For older systems
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

# For CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Memory Issues

**Problem**: Out of memory during training
```bash
RuntimeError: CUDA out of memory
```

**Solutions**:
```bash
# Reduce batch size
python train_model.py --batch-size 1

# Use smaller model
python train_model.py --d-model 256 --n-layers 4

# Use gradient checkpointing
python train_model.py --gradient-checkpointing

# Use CPU instead of GPU
python train_model.py --device cpu
```

#### 3. Training Instability

**Problem**: Loss explodes or becomes NaN
```bash
Epoch 5: Train Loss: nan, Val Loss: nan
```

**Solutions**:
```bash
# Reduce learning rate
python train_model.py --learning-rate 1e-5

# Add gradient clipping
python train_model.py --grad-clip 0.5

# Use mixed precision training
python train_model.py --mixed-precision
```

#### 4. Slow Training

**Problem**: Training takes too long
```bash
Epoch 1/20: 2 hours remaining
```

**Solutions**:
```bash
# Use smaller dataset
python train_model.py --data-size 1000

# Increase batch size (if memory allows)
python train_model.py --batch-size 8

# Use GPU acceleration
python train_model.py --device cuda

# Reduce model size
python train_model.py --d-model 256 --n-layers 4
```

#### 5. Poor Generation Quality

**Problem**: Generated scripts are incorrect or incomplete

**Solutions**:
```bash
# Train longer
python train_model.py --epochs 40

# Use more training data
python train_model.py --data-size 10000

# Increase model size
python train_model.py --d-model 768 --n-layers 8

# Improve data quality
python data_generator.py --quality-filter
```

#### 6. Validation Loss Not Improving

**Problem**: Model overfitting to training data
```bash
Epoch 10: Train Loss: 1.2, Val Loss: 3.8 (increasing)
```

**Solutions**:
```bash
# Add regularization
python train_model.py --weight-decay 0.1 --dropout 0.2

# Use early stopping
python train_model.py --early-stopping --patience 5

# Increase validation data
python train_model.py --val-split 0.3

# Use data augmentation
python train_model.py --data-augmentation
```

### Debug Mode

#### Enable Detailed Logging
```python
# Add to train_model.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable PyTorch debugging
torch.autograd.set_detect_anomaly(True)
```

#### Check Model Architecture
```python
# Verify model structure
python -c "
from model import ManimLLM, ManimLLMConfig
config = ManimLLMConfig(vocab_size=8000)
model = ManimLLM(**config.to_dict())
print(model)
print(f'Parameters: {model.get_model_size():,}')
"
```

#### Test Data Pipeline
```python
# Verify data loading
python -c "
from trainer import create_data_loaders
from tokenizer import ManimTokenizer
import json

# Load data
with open('train_data.json') as f:
    train_data = json.load(f)[:100]  # Test with small subset

# Create tokenizer
tokenizer = ManimTokenizer()
tokenizer.load('tokenizer.pkl')

# Create data loader
train_loader, _ = create_data_loaders(train_data, [], tokenizer, batch_size=2)

# Test batch
for batch in train_loader:
    print('Batch shape:', batch['input_ids'].shape)
    print('Sample tokens:', batch['input_ids'][0][:20])
    break
"
```

---

## Performance Optimization

### Hardware Optimization

#### GPU Optimization
```python
# GPU settings for optimal performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable cuDNN optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

#### CPU Optimization
```python
# CPU settings
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Use optimized CPU operations
torch.set_num_threads(8)
```

### Training Optimization

#### Mixed Precision Training
```python
# Enable automatic mixed precision
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# Training loop with AMP
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Gradient Accumulation
```python
# Simulate larger batch size
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

for i, batch in enumerate(train_loader):
    outputs = model(batch['input_ids'])
    loss = criterion(outputs, batch['target_ids'])
    loss = loss / accumulation_steps  # Normalize
    
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Data Loading Optimization
```python
# Optimize data loading
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,        # Parallel loading
    pin_memory=True,      # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2     # Prefetch batches
)
```

### Memory Optimization

#### Gradient Checkpointing
```python
# Trade compute for memory
def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing to save memory"""
    for block in model.transformer_blocks:
        block.gradient_checkpointing = True
```

#### Model Parallelism
```python
# Split model across GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## Usage Examples

### Basic Usage After Training

#### Simple Script Generation
```python
from agent import ManimAgent

# Initialize with your trained model
agent = ManimAgent(llm_provider="custom")

# Generate basic animations
requests = [
    "Create a red circle",
    "Make a blue square that moves right",
    "Show the text 'Hello Manim'",
    "Display the equation E=mc²"
]

for request in requests:
    print(f"Request: {request}")
    script = agent.generate_script(request)
    print("Generated Script:")
    print("-" * 50)
    print(script)
    print("=" * 60)
```

#### Batch Processing
```python
# Process multiple requests efficiently
from agent import ManimAgent
import json

agent = ManimAgent(llm_provider="custom")

# Load requests from file
with open('animation_requests.json', 'r') as f:
    requests = json.load(f)

# Generate all scripts
results = []
for i, request in enumerate(requests):
    print(f"Processing {i+1}/{len(requests)}: {request}")
    script = agent.generate_script(request)
    
    results.append({
        'id': i,
        'request': request,
        'script': script,
        'timestamp': datetime.now().isoformat()
    })

# Save results
with open('generated_animations.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Generated {len(results)} animation scripts")
```

### Advanced Usage

#### Interactive Script Generator
```python
# Interactive mode with validation
from agent import ManimAgent
from validator import ManimScriptValidator

agent = ManimAgent(llm_provider="custom")
validator = ManimScriptValidator()

def interactive_generator():
    print("🎬 Interactive Manim Script Generator")
    print("Enter your animation requests (type 'quit' to exit)")
    
    while True:
        request = input("\n📝 Request: ").strip()
        
        if request.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not request:
            continue
        
        print("🤔 Generating script...")
        script = agent.generate_script(request)
        
        print("🔍 Validating script...")
        is_valid, fixed_script, report = validator.validate_and_fix(script)
        
        print("\n" + "="*60)
        print("📜 GENERATED SCRIPT:")
        print("="*60)
        print(fixed_script if is_valid else script)
        
        print("\n" + "="*60)
        print("📊 VALIDATION REPORT:")
        print("="*60)
        print(report)
        
        # Option to save
        save = input("\n💾 Save script? (y/n): ").strip().lower()
        if save in ['y', 'yes']:
            filename = input("📁 Filename (without .py): ").strip()
            if filename:
                with open(f"{filename}.py", 'w') as f:
                    f.write(fixed_script if is_valid else script)
                print(f"✅ Saved to {filename}.py")

# Run interactive generator
interactive_generator()
```

#### Web Interface
```python
# Simple web interface using Flask
from flask import Flask, request, jsonify, render_template_string
from agent import ManimAgent

app = Flask(__name__)
agent = ManimAgent(llm_provider="custom")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Manim Script Generator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .script { background: #fff; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Manim Script Generator</h1>
        <p>Enter your animation request below:</p>
        
        <textarea id="request" placeholder="e.g., Create a blue circle that rotates"></textarea>
        <br>
        <button onclick="generateScript()">Generate Script</button>
        
        <div id="result" style="margin-top: 20px;"></div>
    </div>

    <script>
        async function generateScript() {
            const request = document.getElementById('request').value;
            if (!request.trim()) {
                alert('Please enter a request');
                return;
            }
            
            document.getElementById('result').innerHTML = '<p>🤔 Generating script...</p>';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ request: request })
                });
                
                const data = await response.json();
                
                document.getElementById('result').innerHTML = `
                    <h3>📜 Generated Script:</h3>
                    <div class="script">${data.script}</div>
                    <br>
                    <button onclick="downloadScript('${data.script}')">📥 Download Script</button>
                `;
            } catch (error) {
                document.getElementById('result').innerHTML = `<p style="color: red;">❌ Error: ${error.message}</p>`;
            }
        }
        
        function downloadScript(script) {
            const blob = new Blob([script], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'manim_script.py';
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        user_request = data.get('request', '')
        
        if not user_request:
            return jsonify({'error': 'No request provided'}), 400
        
        script = agent.generate_script(user_request)
        
        return jsonify({
            'script': script,
            'request': user_request
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Manim Script Generator Web Interface")
    print("📍 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
```

### Integration Examples

#### Jupyter Notebook Integration
```python
# Use in Jupyter notebooks
from IPython.display import display, Code
from agent import ManimAgent

agent = ManimAgent(llm_provider="custom")

def generate_and_display(request):
    """Generate script and display in notebook"""
    script = agent.generate_script(request)
    display(Code(script, language='python'))
    return script

# Usage in notebook
script = generate_and_display("Create a sine wave animation")
```

#### Command Line Tool
```python
#!/usr/bin/env python3
# manim_generate.py - Command line script generator

import argparse
import sys
from agent import ManimAgent

def main():
    parser = argparse.ArgumentParser(description='Generate Manim scripts from natural language')
    parser.add_argument('request', help='Animation request')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-v', '--validate', action='store_true', help='Validate generated script')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ManimAgent(llm_provider="custom")
    
    # Generate script
    script = agent.generate_script(args.request)
    
    # Validate if requested
    if args.validate:
        from validator import ManimScriptValidator
        validator = ManimScriptValidator()
        is_valid, fixed_script, report = validator.validate_and_fix(script)
        
        if is_valid:
            print("✅ Script is valid")
            script = fixed_script
        else:
            print("⚠️  Script has issues:")
            print(report)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(script)
        print(f"📁 Script saved to {args.output}")
    else:
        print(script)

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
# Generate script to stdout
python manim_generate.py "Create a bouncing ball"

# Save to file
python manim_generate.py "Create a sine wave" -o sine_wave.py

# Generate and validate
python manim_generate.py "Create a complex animation" -v -o animation.py
```

---

## FAQ

### General Questions

**Q: How long does training take?**
A: Training time depends on configuration:
- Small model (5M params): 15-30 minutes
- Medium model (25M params): 45-90 minutes  
- Large model (100M params): 2-4 hours

**Q: Do I need a GPU?**
A: No, but it's recommended. CPU training works but is slower. GPU can speed up training 3-5x.

**Q: How much data do I need?**
A: The system generates synthetic data automatically. 3000 samples work well, but more data (5000-10000) improves quality.

**Q: Can I add my own training examples?**
A: Yes! Edit `data_generator.py` to add custom examples to the `CUSTOM_EXAMPLES` dictionary.

### Technical Questions

**Q: What's the model architecture?**
A: Decoder-only transformer with multi-head attention, similar to GPT but specialized for Manim.

**Q: How does the tokenizer work?**
A: Custom tokenizer that understands Python syntax, Manim functions, and mathematical expressions.

**Q: Can I use pre-trained models?**
A: Currently no, but you can initialize from checkpoints or implement transfer learning.

**Q: How do I improve generation quality?**
A: 
- Train longer (more epochs)
- Use more training data
- Increase model size
- Add custom high-quality examples

### Usage Questions

**Q: How do I use the trained model?**
A: 
```python
from agent import ManimAgent
agent = ManimAgent(llm_provider="custom")
script = agent.generate_script("Your request here")
```

**Q: Can I run multiple models?**
A: Yes, specify different model paths:
```python
agent = ManimAgent(llm_provider="custom", 
                   model_path="model1.pth", 
                   tokenizer_path="tokenizer1.pkl")
```

**Q: How do I validate generated scripts?**
A: Use the built-in validator:
```python
from validator import ManimScriptValidator
validator = ManimScriptValidator()
is_valid, fixed_script, report = validator.validate_and_fix(script)
```

### Troubleshooting Questions

**Q: Training fails with "CUDA out of memory"?**
A: Reduce batch size: `--batch-size 1` or use CPU: `--device cpu`

**Q: Generated scripts are poor quality?**
A: Train longer, use more data, or increase model size.

**Q: Training loss becomes NaN?**
A: Reduce learning rate: `--learning-rate 1e-5` or add gradient clipping.

**Q: Model doesn't improve after many epochs?**
A: You may be overfitting. Use more data, add regularization, or reduce model size.

---
### Resume Training
```sh
    python train_model.py --resume best_model_epoch_1.pth
```

## Conclusion

This comprehensive training guide provides everything needed to create your own custom Manim LLM. The system is designed to be:

- **Accessible**: Works on standard hardware
- **Flexible**: Highly configurable for different needs
- **Extensible**: Easy to add custom data and examples
- **Practical**: Generates real, working Manim scripts

### Next Steps

1. **Set up environment**: Install PyTorch and dependencies
2. **Run system test**: Verify all components work
3. **Start training**: Begin with default settings
4. **Evaluate results**: Test generated scripts
5. **Iterate**: Improve with more data or larger models

### Support and Resources

- **Documentation**: This guide and inline code comments
- **Examples**: Comprehensive examples in `examples.py`
- **Testing**: System tests in `test_system.py`
- **Evaluation**: Built-in evaluation tools

The custom Manim LLM system represents a complete, production-ready solution for automated animation script generation. With this guide, you have all the tools needed to create, train, and deploy your own specialized language model.

Happy training! 🚀
