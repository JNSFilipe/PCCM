import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torchmetrics as tm
from enum import Enum, auto
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
            nn.Conv1d(in_channels=5, out_channels=32,
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


class Reporter:
    def __init__(self, base_path, model, optim, lr, device="cpu"):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.epochs = []

        # Initialize metrics
        self.accuracy = tm.classification.Accuracy(task="binary").to(device)
        self.precision = tm.classification.Precision(task="binary").to(device)
        self.recall = tm.classification.Recall(task="binary").to(device)
        self.f1 = tm.classification.F1Score(task="binary").to(device)

        # Create unique path for this run
        cname = type(model).__name__
        oname = type(optim).__name__
        self.path = f"{
            base_path}/train_{cname}_{oname}_{lr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(self.path).mkdir(parents=True, exist_ok=True)

        # Initialize the plot
        self.fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Loss', 'Model Accuracy'),
            horizontal_spacing=0.15
        )

        # Add traces that we'll update
        self.fig.add_trace(go.Scatter(
            name="Train Loss", mode='lines'), row=1, col=1)
        self.fig.add_trace(go.Scatter(
            name="Val Loss", mode='lines'), row=1, col=1)
        self.fig.add_trace(go.Scatter(
            name="Train Accuracy", mode='lines'), row=1, col=2)
        self.fig.add_trace(go.Scatter(
            name="Val Accuracy", mode='lines'), row=1, col=2)

        # Update layout
        self.fig.update_layout(
            height=500,
            width=1200,
            showlegend=True,
            title_text="Training Progress",
            title_x=0.5,
            template="plotly_white"
        )

        # Update axes
        self.fig.update_xaxes(title_text="Epoch", row=1, col=1)
        self.fig.update_xaxes(title_text="Epoch", row=1, col=2)
        self.fig.update_yaxes(title_text="Loss", row=1, col=1)
        self.fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        # Update the plot
        self.fig.data[0].x = self.epochs
        self.fig.data[0].y = self.train_losses
        self.fig.data[1].x = self.epochs
        self.fig.data[1].y = self.val_losses
        self.fig.data[2].x = self.epochs
        self.fig.data[2].y = self.train_accuracies
        self.fig.data[3].x = self.epochs
        self.fig.data[3].y = self.val_accuracies

        # Save the plot
        self.fig.write_html(f"{self.path}/training_progress.html")

    def save_model(self, model, is_best=False):
        path = f"{self.path}/checkpoints/"
        Path(path).mkdir(parents=True, exist_ok=True)
        if is_best:
            torch.save(model.state_dict(), f"{path}/best_model.pth")
        else:
            torch.save(model.state_dict(), f"{path}/latest_model.pth")

    def compute_metrics(self, y_pred, y_true):
        accuracy = self.accuracy(y_pred, y_true)
        precision = self.precision(y_pred, y_true)
        recall = self.recall(y_pred, y_true)
        f1 = self.f1(y_pred, y_true)
        return accuracy, precision, recall, f1


class StockPredictor:
    def __init__(self):
        # Training parameters
        self.num_epochs = 100
        self.batch_size = 32
        self.lr = 0.001
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model, optimizer, and loss
        self.model = StockPriceCNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Initialize reporter
        self.reporter = Reporter(
            "stock_prediction", self.model, self.optimizer, self.lr, self.device
        )

        self.best_val_loss = float('inf')

    def load_and_balance_data(self, data_dir):
        """Load and combine data from all tickers, ensuring balanced classes."""
        all_sequences = []
        all_labels = []

        # Load all data
        for filename in os.listdir(data_dir):
            if filename.endswith('_sequences.npy'):
                ticker = filename.replace('_sequences.npy', '')
                sequences = np.load(os.path.join(
                    data_dir, f'{ticker}_sequences.npy'))
                labels = np.load(os.path.join(
                    data_dir, f'{ticker}_labels.npy'))
                all_sequences.append(sequences)
                all_labels.append(labels)

        # Combine all data
        X = np.concatenate(all_sequences)
        y = np.concatenate(all_labels)

        # Balance classes
        up_indices = np.where(y == 1)[0]
        down_indices = np.where(y == 0)[0]
        min_class_size = min(len(up_indices), len(down_indices))

        if len(up_indices) > min_class_size:
            up_indices = np.random.choice(
                up_indices, min_class_size, replace=False)
        if len(down_indices) > min_class_size:
            down_indices = np.random.choice(
                down_indices, min_class_size, replace=False)

        balanced_indices = np.concatenate([up_indices, down_indices])
        np.random.shuffle(balanced_indices)

        return X[balanced_indices], y[balanced_indices]

    def initialize_dataloaders(self, sequences, labels):
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device == "cuda"),
            num_workers=4
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == "cuda"),
            num_workers=4
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []

        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions.extend(output.argmax(dim=1).cpu())
            targets.extend(target.cpu())

        avg_loss = total_loss / len(self.train_loader)
        accuracy, _, _, _ = self.reporter.compute_metrics(
            torch.tensor(predictions), torch.tensor(targets)
        )

        return avg_loss, accuracy.item()

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                predictions.extend(output.argmax(dim=1).cpu())
                targets.extend(target.cpu())

        avg_loss = total_loss / len(self.val_loader)
        accuracy, precision, recall, f1 = self.reporter.compute_metrics(
            torch.tensor(predictions), torch.tensor(targets)
        )

        print(f"\nValidation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return avg_loss, accuracy.item()

    def run(self):
        print("Loading and preparing data...")
        sequences, labels = self.load_and_balance_data('ml_data')
        self.initialize_dataloaders(sequences, labels)

        print(f"Starting training on {self.device}...")
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")

            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            # Update learning rate
            self.lr_scheduler.step(val_loss)

            # Update plots and save progress
            self.reporter.update(
                epoch, train_loss, train_acc, val_loss, val_acc)

            # Save model if it's the best so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.reporter.save_model(self.model, is_best=True)

            # Also save latest model
            self.reporter.save_model(self.model, is_best=False)


if __name__ == "__main__":
    predictor = StockPredictor()
    predictor.run()
