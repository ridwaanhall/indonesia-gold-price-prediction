import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import GoldPriceLSTM
from data_loader import get_data_loader

# Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
SEQUENCE_LENGTH = 30
INPUT_SIZE = 4  # Features from DataLoader

# Initialize model
model = GoldPriceLSTM(INPUT_SIZE)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load dataset
file_path = "data/processed/gold_prices_cleaned.csv"
data_loader = get_data_loader(file_path, BATCH_SIZE, SEQUENCE_LENGTH)

# Training loop
def train(model, data_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()

            # Forward pass
            predictions = model(inputs).squeeze()
            loss = criterion(predictions, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")
    
    # Save trained model
    os.makedirs("models", exist_ok=True)  # Create models directory if it doesn't exist
    torch.save(model.state_dict(), "models/gold_price_lstm.pth")
    print("✅ Model training complete & saved!")

train(model, data_loader, criterion, optimizer, EPOCHS)
