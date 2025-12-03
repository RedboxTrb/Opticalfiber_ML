"""ML models for optical signal equalization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleCNN(nn.Module):
    """Simple CNN equalizer."""

    def __init__(self, input_length=128, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)

        # Calculate flatten size
        self.flatten_size = 128 * (input_length // 8)

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch, seq_len) -> (batch, 1, seq_len)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEqualizer(nn.Module):
    """Transformer-based equalizer (NOVEL for optical comms!)."""

    def __init__(self, input_length=128, num_classes=2, d_model=64, nhead=4, num_layers=3):
        super().__init__()

        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        # Project to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        x = self.classifier(x)

        return x


class BiLSTMEqualizer(nn.Module):
    """Bidirectional LSTM equalizer for temporal sequence processing."""

    def __init__(self, input_length=128, num_classes=2, hidden_size=128, num_layers=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)

        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size*2)

        # Classification
        output = self.classifier(context)

        return output


class CNNLSTMHybrid(nn.Module):
    """CNN-LSTM Hybrid: CNN for feature extraction + LSTM for temporal modeling."""

    def __init__(self, input_length=128, num_classes=2):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(0.2)

        # LSTM for temporal processing
        # After 3 pooling layers: input_length // 8
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Prepare for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)

        # Global average pooling over time
        x = lstm_out.mean(dim=1)  # (batch, hidden_size*2)

        # Classification
        x = self.classifier(x)

        return x


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network (TCN) with dilated convolutions."""

    def __init__(self, input_length=128, num_classes=2, num_channels=[64, 128, 256], kernel_size=3):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            # Dilated causal convolution
            layers.append(nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size - 1) * dilation_size,
                dilation=dilation_size
            ))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

            # Residual connection
            if i > 0:
                layers.append(ResidualBlock(num_channels[i-1], out_channels))

        self.network = nn.Sequential(*layers)

        # Global pooling and classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)

        # TCN processing
        x = self.network(x)

        # Trim to original length (remove padding)
        x = x[:, :, :x.size(2)]

        # Classification
        x = self.classifier(x)

        return x


class ResidualBlock(nn.Module):
    """Residual connection for TCN."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        if self.conv:
            return self.conv(x)
        return x


class AttentionCNN(nn.Module):
    """CNN with Channel and Spatial Attention (CBAM-inspired)."""

    def __init__(self, input_length=128, num_classes=2):
        super().__init__()

        # Convolutional layers with attention
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.attention1 = ChannelAttention(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.attention2 = ChannelAttention(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.attention3 = ChannelAttention(256)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)

        # Calculate flatten size
        self.flatten_size = 256 * (input_length // 8)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Layer 1 with attention
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x) * x
        x = self.pool(x)
        x = self.dropout(x)

        # Layer 2 with attention
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x) * x
        x = self.pool(x)
        x = self.dropout(x)

        # Layer 3 with attention
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x) * x
        x = self.pool(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class ChannelAttention(nn.Module):
    """Channel attention mechanism."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        b, c, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        out = torch.sigmoid(avg_out + max_out).view(b, c, 1)
        return out


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda'):
    """Quick training function."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_acc = correct / total
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}')

    return train_losses, val_accs


def evaluate_ber(model, test_loader, device='cuda'):
    """Evaluate Bit Error Rate."""
    model.eval()
    errors = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)

            errors += (predicted != y_batch).sum().item()
            total += y_batch.size(0)

    ber = errors / total if total > 0 else 0
    return ber
