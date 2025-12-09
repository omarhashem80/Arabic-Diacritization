import torch
import torch.nn as nn
import json


class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 dropout_rate, num_layers=1, max_length=600):
        super(CharBiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )
        self.batch_norm = nn.BatchNorm1d(max_length)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x_embed = self.embedding(x)
        lstm_out, _ = self.lstm(x_embed)
        lstm_out = self.batch_norm(lstm_out)
        out = self.fc(lstm_out)
        return out

    @staticmethod
    def load_model(model_file="best_model.pth", meta_file="best_model_meta.json", device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_path = '/kaggle/working/'

        with open(base_path + meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model = CharBiLSTM(
            vocab_size=metadata["vocab_size"],
            embedding_dim=metadata["embedding_size"],
            hidden_dim=metadata["hidden_size"],
            output_dim=metadata["output_classes"],
            dropout_rate=metadata["dropout_rate"],
            num_layers=metadata["num_layers"],
            max_length=metadata["max_sequence_length"]
        ).to(device)

        # ---- Load best weights ----
        state_dict = torch.load(base_path + model_file, map_location=device)
        model.load_state_dict(state_dict)

        model.eval()
        return model, metadata