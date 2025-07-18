#!/usr/bin/env python3
"""
Training Script for Custom Manim LLM
Run this script to train your own Manim script generation model.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generator import ManimDataGenerator
from tokenizer import ManimTokenizer, build_tokenizer_from_data
from model import ManimLLM, ManimLLMConfig
from trainer import ManimTrainer, create_data_loaders

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Custom Manim LLM')
    
    # Data arguments
    parser.add_argument('--data-size', type=int, default=3000,
                        help='Number of training samples to generate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=2048,
                        help='Feed-forward dimension')
    parser.add_argument('--max-len', type=int, default=1024,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length for training')
    
    # File arguments
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Directory for data files')
    parser.add_argument('--model-dir', type=str, default='.',
                        help='Directory for model files')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()

def setup_environment(args):
    """Setup training environment."""
    import torch
    import random
    import numpy as np
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def prepare_data(args):
    """Prepare training data."""
    train_file = os.path.join(args.data_dir, 'train_data.json')
    val_file = os.path.join(args.data_dir, 'val_data.json')
    
    if os.path.exists(train_file) and os.path.exists(val_file):
        print("Loading existing training data...")
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(val_file, 'r') as f:
            val_data = json.load(f)
    else:
        print(f"Generating {args.data_size} training samples...")
        generator = ManimDataGenerator()
        data = generator.generate_and_save_dataset(
            num_samples=args.data_size,
            filename=os.path.join(args.data_dir, 'full_data.json')
        )
        
        # Split data
        train_data, val_data = generator.create_validation_split(
            data, split_ratio=1-args.val_split
        )
        
        # Save splits
        generator.save_training_data(train_data, train_file)
        generator.save_training_data(val_data, val_file)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

def prepare_tokenizer(args, train_data, val_data):
    """Prepare tokenizer."""
    tokenizer_file = os.path.join(args.model_dir, 'tokenizer.pkl')
    
    if os.path.exists(tokenizer_file):
        print("Loading existing tokenizer...")
        tokenizer = ManimTokenizer()
        tokenizer.load(tokenizer_file)
    else:
        print("Building tokenizer...")
        tokenizer = build_tokenizer_from_data(
            train_data + val_data, 
            vocab_size=args.vocab_size
        )
        tokenizer.save(tokenizer_file)
    
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer

def create_model(args, tokenizer):
    """Create model."""
    config = ManimLLMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len
    )
    
    model = ManimLLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_len=config.max_len
    )
    
    print(f"Model created with {model.get_model_size():,} parameters")
    return model, config

def train_model(args, model, tokenizer, train_data, val_data):
    """Train the model."""
    # Create training configuration
    training_config = {
        'vocab_size': tokenizer.get_vocab_size(),
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_ff,
        'max_len': args.max_len,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'max_length': args.max_length
    }
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Create trainer
    trainer = ManimTrainer(model, tokenizer, training_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")
    
    # Train model
    start_time = time.time()
    trainer.train(train_loader, val_loader, args.epochs)
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_manim_model.pth')
    trainer.save_checkpoint(final_model_path, args.epochs, trainer.best_val_loss)
    
    # Save training history plot
    history_plot_path = os.path.join(args.model_dir, 'training_history.png')
    trainer.plot_training_history(history_plot_path)
    
    return trainer

def evaluate_model(args, trainer, tokenizer):
    """Evaluate the trained model."""
    print("\nEvaluating model...")
    
    # Test requests
    test_requests = [
        "Create a blue circle",
        "Make a red square that moves to the right",
        "Show the mathematical formula E=mc²",
        "Create a text that says 'Hello World'",
        "Draw a coordinate system with axes"
    ]
    
    # Load best model for evaluation
    if trainer.best_model_path:
        print(f"Loading best model: {trainer.best_model_path}")
        trainer.model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=trainer.device)['model_state_dict']
        )
    
    # Generate sample outputs
    from inference import ManimInferenceEngine
    
    model_path = os.path.join(args.model_dir, trainer.best_model_path or 'final_manim_model.pth')
    tokenizer_path = os.path.join(args.model_dir, 'tokenizer.pkl')
    
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        engine = ManimInferenceEngine(model_path, tokenizer_path)
        
        results_file = os.path.join(args.model_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("MANIM LLM EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for i, request in enumerate(test_requests):
                print(f"Testing: {request}")
                result = engine.generate_and_validate(request)
                
                f.write(f"Request {i+1}: {request}\n")
                f.write(f"Score: {result['best_score']:.1f}\n")
                f.write("-" * 30 + "\n")
                f.write(result['best_script'])
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"Evaluation results saved to {results_file}")
    else:
        print("Cannot evaluate: model or tokenizer files not found")

def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("CUSTOM MANIM LLM TRAINING")
    print("=" * 60)
    print(f"Data size: {args.data_size}")
    print(f"Model size: {args.d_model}d, {args.n_layers}L, {args.n_heads}H")
    print(f"Training: {args.epochs} epochs, batch size {args.batch_size}")
    print("=" * 60)
    
    # Setup environment
    device = setup_environment(args)
    
    # Prepare data
    train_data, val_data = prepare_data(args)
    
    # Prepare tokenizer
    tokenizer = prepare_tokenizer(args, train_data, val_data)
    
    # Create model
    model, config = create_model(args, tokenizer)
    
    # Train model
    trainer = train_model(args, model, tokenizer, train_data, val_data)
    
    # Evaluate model
    evaluate_model(args, trainer, tokenizer)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: {args.model_dir}")
    print(f"To use your model:")
    print(f"  from agent import ManimAgent")
    print(f"  agent = ManimAgent('custom')")
    print(f"  script = agent.generate_script('Create a blue circle')")
    print("=" * 60)

if __name__ == "__main__":
    main()