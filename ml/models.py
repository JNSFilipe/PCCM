import torch
import torch.nn as nn


class StockPriceIndicatorsCNN(nn.Module):
    def __init__(self, sequence_length=252):
        super(StockPriceIndicatorsCNN, self).__init__()

        # Input shape: [batch_size, channels=9, sequence_length=252]
        # Channels: Open, High, Low, Close, Volume, RSI, MACD, MACD_Signal, MACD_Diff

        # Split the input features into price/volume features and technical indicators
        self.price_vol_conv = nn.Sequential(
            # Process OHLCV data (5 channels)
            nn.Conv1d(in_channels=5, out_channels=32,
                      kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )

        self.tech_ind_conv = nn.Sequential(
            # Process technical indicators (4 channels)
            nn.Conv1d(in_channels=10, out_channels=16,
                      kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )

        # Combined CNN layers after concatenation
        self.combined_conv = nn.Sequential(
            # Input channels = 32 (price_vol) + 16 (tech_ind) = 48
            nn.Conv1d(in_channels=48, out_channels=64,
                      kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )

        # Calculate the size of flattened features
        # Sequence length is divided by 8 due to 3 max pooling layers (2 * 2 * 2)
        self.feature_size = 128 * (sequence_length // (2 * 2 * 2))

        # Fully connected layers
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
        # Split input into price/volume data and technical indicators
        price_vol_data = x[:, :5, :]  # First 5 channels (OHLCV)
        tech_ind_data = x[:, 5:, :]   # Last 4 channels (RSI, MACD, etc.)

        # Process each data stream separately
        price_vol_features = self.price_vol_conv(price_vol_data)
        tech_ind_features = self.tech_ind_conv(tech_ind_data)

        # Concatenate features along the channel dimension
        combined = torch.cat([price_vol_features, tech_ind_features], dim=1)

        # Process combined features
        x = self.combined_conv(combined)

        # Flatten and pass through fully connected layers
        x = x.view(-1, self.feature_size)
        x = self.fc_layers(x)

        return x


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
