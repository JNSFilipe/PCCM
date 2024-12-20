import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


class StockDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

        # Transpose to [batch, channels, sequence_length]
        self.sequences = self.sequences.transpose(1, 2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class StockPriceCNN(nn.Module):
    def __init__(self):
        super(StockPriceCNN, self).__init__()

        # Input shape: [batch_size, channels=5, sequence_length=252]
        self.cnn_layers = nn.Sequential(
            # First CNN block
            # nn.Conv1d(in_channels=5, out_channels=32,
            #           kernel_size=7, padding=3),
            nn.Conv1d(in_channels=21, out_channels=32,
                      kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),

            # Second CNN block
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),

            # Third CNN block
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )

        # Calculate the size of flattened features
        # Divided by 8 due to 3 max pooling layers
        self.feature_size = 128 * (252 // (2 * 2 * 2))

        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)  # Binary classification
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, self.feature_size)
        x = self.fc_layers(x)
        return x


def load_and_balance_data(data_dir):
    """Load and combine data from all tickers, ensuring balanced classes."""
    all_sequences = []
    all_labels = []

    # Load all data
    for filename in os.listdir(data_dir):
        if filename.endswith('_sequences.npy'):
            ticker = filename.replace('_sequences.npy', '')

            sequences = np.load(os.path.join(
                data_dir, f'{ticker}_sequences.npy'))
            labels = np.load(os.path.join(data_dir, f'{ticker}_labels.npy'))

            all_sequences.append(sequences)
            all_labels.append(labels)

    # Combine all data
    X = np.concatenate(all_sequences)
    y = np.concatenate(all_labels)

    # Get indices for each class
    up_indices = np.where(y == 1)[0]
    down_indices = np.where(y == 0)[0]

    # Find minimum class size
    min_class_size = min(len(up_indices), len(down_indices))

    # Randomly sample from larger class to match smaller class
    if len(up_indices) > min_class_size:
        up_indices = np.random.choice(
            up_indices, min_class_size, replace=False)
    if len(down_indices) > min_class_size:
        down_indices = np.random.choice(
            down_indices, min_class_size, replace=False)

    # Combine balanced indices
    balanced_indices = np.concatenate([up_indices, down_indices])
    np.random.shuffle(balanced_indices)

    # Return balanced dataset
    return X[balanced_indices], y[balanced_indices]


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Train the model and return training history."""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_true = []

        for sequences, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(labels.cpu().numpy())

        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_true = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        # Calculate metrics
        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_true, train_preds)
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_true, val_preds)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.4f}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    return history


def plot_training_history(history):
    """Plot training and validation metrics using Plotly."""
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Loss', 'Model Accuracy'),
        horizontal_spacing=0.15
    )

    # Add traces for loss
    fig.add_trace(
        go.Scatter(y=history['train_loss'], name="Train Loss",
                   line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history['val_loss'], name="Validation Loss",
                   line=dict(color="red")),
        row=1, col=1
    )

    # Add traces for accuracy
    fig.add_trace(
        go.Scatter(y=history['train_acc'], name="Train Accuracy",
                   line=dict(color="green")),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=history['val_acc'], name="Validation Accuracy",
                   line=dict(color="orange")),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        height=500,
        width=1200,
        showlegend=True,
        title_text="Training History",
        title_x=0.5,
        template="plotly_white"
    )

    # Update x-axes
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)

    # Update y-axes
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    # Save as HTML file
    fig.write_html("training_history.html")

    # Optional: Also display in notebook if running in one
    fig.show()


def compute_class_weights(labels):
    """Compute class weights for balanced training."""
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters
    data_dir = 'ml_data'
    # batch_size = 32
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load balanced data
    print("Loading and balancing data...")
    sequences, labels = load_and_balance_data(data_dir)

    # Compute class weights for balanced training
    class_weights = compute_class_weights(labels).to(device)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create data loaders
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, optimizer, and scheduler
    model = StockPriceCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    # Train model
    print("Starting training...")
    history = train_model(model, train_loader, val_loader,
                          criterion, optimizer, scheduler, num_epochs, device)

    # Plot training history
    plot_training_history(history)

    # Load best model and evaluate on validation set
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    val_preds = []
    val_true = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.numpy())

    # Print final metrics
    print("\nFinal Validation Metrics:")
    print(f"Accuracy: {accuracy_score(val_true, val_preds):.4f}")
    print(f"Precision: {precision_score(val_true, val_preds):.4f}")
    print(f"Recall: {recall_score(val_true, val_preds):.4f}")
    print(f"F1 Score: {f1_score(val_true, val_preds):.4f}")


if __name__ == "__main__":
    main()
