"""
Training Pipeline for Custom Manim LLM
Handles model training, evaluation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import ManimLLM, ManimLLMConfig
from tokenizer import ManimTokenizer, build_tokenizer_from_data
from data_generator import ManimDataGenerator

class ManimDataset(Dataset):
    """Dataset for Manim script generation."""
    
    def __init__(self, data: List[Dict], tokenizer: ManimTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Create request-response pair
        request = sample['request']
        script = sample['script']
        
        # Encode the full sequence
        full_sequence = self.tokenizer.create_request_response_pair(request, script)
        
        # Truncate if too long
        if len(full_sequence) > self.max_length:
            full_sequence = full_sequence[:self.max_length]
        
        # Create input and target sequences
        input_ids = full_sequence[:-1]  # All tokens except last
        target_ids = full_sequence[1:]  # All tokens except first
        
        # Pad sequences
        input_ids = self._pad_sequence(input_ids, self.max_length - 1)
        target_ids = self._pad_sequence(target_ids, self.max_length - 1)
        
        # Create attention mask
        attention_mask = [1 if token_id != self.tokenizer.vocab['<PAD>'] else 0 for token_id in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def _pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """Pad sequence to max_length."""
        if len(sequence) < max_length:
            sequence.extend([self.tokenizer.vocab['<PAD>']] * (max_length - len(sequence)))
        return sequence[:max_length]

class ManimTrainer:
    """Trainer for Manim LLM."""
    
    def __init__(self, model: ManimLLM, tokenizer: ManimTokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 10),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<PAD>'])
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_tokens += attention_mask.sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating")
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
                
                # Update metrics
                total_loss += loss.item()
                total_tokens += attention_mask.sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        """Train the model."""
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_trainable_parameters():,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_perplexity = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_perplexity = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_perplexities.append(train_perplexity)
            self.val_perplexities.append(val_perplexity)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = f"best_model_epoch_{epoch + 1}.pth"
                self.save_checkpoint(self.best_model_path, epoch, val_loss)
                print(f"New best model saved: {self.best_model_path}")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model saved at: {self.best_model_path}")
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_perplexities = checkpoint.get('train_perplexities', [])
        self.val_perplexities = checkpoint.get('val_perplexities', [])
        
        print(f"Checkpoint loaded from {path}")
        print(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    def plot_training_history(self, save_path: str = "training_history.png") -> None:
        """Plot training history."""
        if not self.train_losses:
            print("No training history to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot perplexities
        ax2.plot(self.train_perplexities, label='Train Perplexity', color='blue')
        ax2.plot(self.val_perplexities, label='Val Perplexity', color='red')
        ax2.set_title('Training and Validation Perplexity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.legend()
        ax2.grid(True)
        
        # Plot learning rate
        ax3.plot([self.scheduler.get_last_lr()[0]] * len(self.train_losses))
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Plot loss difference
        loss_diff = [abs(train - val) for train, val in zip(self.train_losses, self.val_losses)]
        ax4.plot(loss_diff, color='green')
        ax4.set_title('Train-Val Loss Difference')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
        print(f"Training history saved to {save_path}")

def create_data_loaders(train_data: List[Dict], val_data: List[Dict], 
                       tokenizer: ManimTokenizer, batch_size: int = 8, 
                       max_length: int = 512) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation."""
    train_dataset = ManimDataset(train_data, tokenizer, max_length)
    val_dataset = ManimDataset(val_data, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    """Main training function."""
    # Configuration
    config = {
        'vocab_size': 8000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_len': 1024,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 4,
        'epochs': 20,
        'max_length': 512
    }
    
    # Generate or load training data
    print("Generating training data...")
    generator = ManimDataGenerator()
    
    # Check if data exists
    if os.path.exists("train_data.json") and os.path.exists("val_data.json"):
        print("Loading existing data...")
        with open("train_data.json", 'r') as f:
            train_data = json.load(f)
        with open("val_data.json", 'r') as f:
            val_data = json.load(f)
    else:
        print("Generating new data...")
        data = generator.generate_and_save_dataset(num_samples=3000)
        train_data, val_data = generator.create_validation_split(data)
        generator.save_training_data(train_data, "train_data.json")
        generator.save_training_data(val_data, "val_data.json")
    
    # Build tokenizer
    print("Building tokenizer...")
    if os.path.exists("tokenizer.pkl"):
        tokenizer = ManimTokenizer()
        tokenizer.load("tokenizer.pkl")
    else:
        tokenizer = build_tokenizer_from_data(train_data + val_data, config['vocab_size'])
        tokenizer.save("tokenizer.pkl")
    
    # Update config with actual vocab size
    config['vocab_size'] = tokenizer.get_vocab_size()
    
    # Create model
    print("Creating model...")
    model_config = ManimLLMConfig(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len']
    )
    
    model = ManimLLM(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        d_ff=model_config.d_ff,
        max_len=model_config.max_len
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )
    
    # Create trainer
    trainer = ManimTrainer(model, tokenizer, config)
    
    # Train model
    print("Starting training...")
    trainer.train(train_loader, val_loader, config['epochs'])
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save final model
    final_model_path = "final_manim_model.pth"
    trainer.save_checkpoint(final_model_path, config['epochs'], trainer.best_val_loss)
    
    print(f"Training completed! Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()