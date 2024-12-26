import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import os
from models import StockPriceCNN, StockPriceIndicatorsCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from cons import K


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


def load_and_balance_data(data_dir):
    """
    Load and combine data from all tickers, ensuring balanced classes across multiple classes.

    Args:
        data_dir (str): Directory containing the sequence and label files

    Returns:
        tuple: (X, y) containing balanced features and labels
    """
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

    # Get unique classes and their counts
    unique_classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

    # Find minimum class size
    min_class_size = min(len(indices) for indices in class_indices.values())

    # Balance all classes to match the smallest class size
    balanced_indices = []
    for cls in unique_classes:
        indices = class_indices[cls]
        if len(indices) > min_class_size:
            indices = np.random.choice(indices, min_class_size, replace=False)
        balanced_indices.append(indices)

    # Combine and shuffle balanced indices
    balanced_indices = np.concatenate(balanced_indices)
    np.random.shuffle(balanced_indices)

    # Print class distribution information
    original_distribution = Counter(y)
    balanced_distribution = Counter(y[balanced_indices])
    print("Original class distribution:", dict(original_distribution))
    print("Balanced class distribution:", dict(balanced_distribution))

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
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.7f}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), K.MDL_DIR)

        # Early stopping (stop if lr < K.MIN_LR*10)
        if optimizer.param_groups[0]['lr'] < K.MIN_LR*10:
            print(f"Early stopping due to lr < {K.MIN_LR*10}")
            break

    return history


def plot_training_history(history, val_true, val_probs):
    """Plot training and validation metrics using Plotly, including confusion matrix and precision-threshold curve."""
    # Create figure with four subplots (2x2 grid)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Loss', 'Model Accuracy',
                        'Confusion Matrix', 'Precision vs Threshold'),
        vertical_spacing=0.15,
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

    # Create confusion matrix
    cm = confusion_matrix(
        val_true, (val_probs >= 0.5).astype(int), normalize="all") * 100

    # Add confusion matrix heatmap
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=[f"Predicted {i}" for i in list(set(val_true))],
            y=[f"Actual {i}" for i in list(set(val_true))],
            text=cm,
            texttemplate="%{z:.1f}%",
            textfont={"size": 12},
            colorscale='Blues',
            showscale=False,
        ),
        row=2, col=1
    )

    # Calculate precision for different thresholds
    thresholds = np.arange(0.5, 0.95, 0.05)
    precisions = []

    for threshold in thresholds:
        predictions = (val_probs >= threshold).astype(int)
        if sum(predictions) > 0:  # Only calculate precision if we have positive predictions
            prec = precision_score(val_true, predictions, zero_division=0)
        else:
            prec = 0
        precisions.append(prec)

    # Add precision vs threshold plot
    fig.add_trace(
        go.Scatter(
            x=thresholds * 100,  # Convert to percentage
            y=precisions,
            name="Precision",
            line=dict(color="purple"),
            mode='lines+markers'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        showlegend=True,
        title_text="Training History and Model Evaluation",
        title_x=0.5,
        template="plotly_white"
    )

    # Update x-axes
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Threshold (%)", row=2, col=2)

    # Update y-axes
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=2, col=2)

    # Save as HTML file
    fig.write_html(K.RPRT_DIR)

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
    torch.manual_seed(K.RANDOM)
    np.random.seed(K.RANDOM)

    # Parameters
    data_dir = K.DATA_DIR
    batch_size = K.BATCH_SIZE
    num_epochs = K.EPOCHS
    learning_rate = K.LEARNING_RATE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load balanced data
    print("Loading and balancing data...")
    sequences, labels = load_and_balance_data(data_dir)

    # Compute class weights for balanced training
    class_weights = compute_class_weights(labels).to(device)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=K.VALIDATION_SPLIT, random_state=K.RANDOM, stratify=labels)

    # Create data loaders
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, optimizer, and scheduler
    model = None
    if not K.INDICATORS:
        model = StockPriceCNN().to(device)
    else:
        model = StockPriceIndicatorsCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=K.PATIENCE,
        verbose=True,
        min_lr=K.MIN_LR
    )

    # Train model
    print("Starting training...")
    history = train_model(model, train_loader, val_loader,
                          criterion, optimizer, scheduler, num_epochs, device)

    # Load best model and evaluate on validation set
    model.load_state_dict(torch.load(K.MDL_DIR))
    model.eval()

    # val_preds = []
    val_probs = []
    val_true = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            # Probability of positive class
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            val_probs.extend(probabilities.cpu().numpy())
            val_true.extend(labels.numpy())

    val_probs = np.array(val_probs)
    val_true = np.array(val_true)

    # Plot training history
    plot_training_history(history, val_true, val_probs)

    val_preds = (val_probs >= 0.5).astype(int)

    # Print final metrics
    print("\nFinal Validation Metrics:")
    print(f"Accuracy: {accuracy_score(val_true, val_preds):.4f}")
    print(f"Precision: {precision_score(val_true, val_preds):.4f}")
    print(f"Recall: {recall_score(val_true, val_preds):.4f}")
    print(f"F1 Score: {f1_score(val_true, val_preds):.4f}")


if __name__ == "__main__":
    main()
