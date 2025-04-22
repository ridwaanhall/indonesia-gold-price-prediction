import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from model import GoldPriceLSTM
from data_loader import get_data_loader

def train_model(config):
    """
    Train the gold price prediction model using the specified configuration.
    
    Args:
        config (dict): Training configuration parameters
    
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
        min_val_loss: Minimum validation loss achieved
    """
    # Set device for training (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, dataset = get_data_loader(
        file_path=config['data_file'],
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        shuffle=True
    )
    
    # Create validation loader with 10% of data
    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.1)
    train_size = dataset_size - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    input_size = len(dataset.features)
    model = GoldPriceLSTM(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_attention=config['use_attention']
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping parameters
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    # Training loop variables
    train_losses = []
    val_losses = []
    
    print(f"\n===== Starting Training =====")
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    print(f"Features: {dataset.features}")
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimize
            optimizer.step()
            
            # Accumulate batch loss
            train_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"\rEpoch {epoch+1}/{config['epochs']} [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.6f}", end="")
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Print epoch results
        print(f"\rEpoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            
            # Save the best model
            os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"✅ Model saved to {config['model_save_path']}")
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_path = f"plots/training_loss_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"✅ Training plot saved to {plot_path}")
    
    # Load the best model for return
    model.load_state_dict(torch.load(config['model_save_path']))
    
    return model, train_losses, val_losses, best_val_loss

if __name__ == "__main__":
    # Training configuration
    config = {
        'data_file': "data/processed/gold_prices_cleaned.csv",
        'model_save_path': "models/gold_price_lstm.pth",
        'batch_size': 32,
        'sequence_length': 60,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'use_attention': True,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 100
    }
    
    print("Starting model training with configuration:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    # Train the model
    model, train_losses, val_losses, best_val_loss = train_model(config)
    
    print(f"\n===== Training Complete =====")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"✅ Model saved at: {config['model_save_path']}")
