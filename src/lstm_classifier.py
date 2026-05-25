# models/lstm_classifier.py
import torch
import torch.nn as nn


class PlantHealthLSTM(nn.Module):
    """
    Main model: Combines plant embedding + sensor time-series
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Plant type embedding
        self.plant_embedding = nn.Embedding(config.num_plants, config.embedding_dim)

        # Sensor feature input size (your sensors + engineered features)
        self.input_size = 12  # Example: 8 raw + 4 engineered

        # LSTM for time-series
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0,
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.lstm_hidden_size + config.embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_health_classes),
        )

    def forward(self, sensor_sequences: torch.Tensor, plant_ids: torch.Tensor):
        # sensor_sequences shape: (batch, sequence_length, num_features)
        # plant_ids shape: (batch,)

        # Get plant embedding
        plant_emb = self.plant_embedding(plant_ids)  # (batch, embedding_dim)

        # Process time-series
        lstm_out, _ = self.lstm(sensor_sequences)  # (batch, seq_len, hidden_size)
        lstm_last = lstm_out[:, -1, :]  # Take last timestep

        # Concatenate plant embedding with LSTM features
        combined = torch.cat([lstm_last, plant_emb], dim=1)

        # Final prediction
        output = self.classifier(combined)  # (batch, 3)

        return output
