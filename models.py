import torch
import torch.nn as nn
import json
import pickle


class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size=44, embedding_dim=300, hidden_dim=256, output_dim=16,
                 dropout_rate=0.2, num_layers=1, max_length=600):
        super(CharBiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )
        self.batchnorm = nn.BatchNorm1d(max_length)
        self.output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x_embed = self.embedding(x)
        lstm_out, _ = self.lstm(x_embed)
        lstm_out = self.batchnorm(lstm_out)
        out = self.output(lstm_out)
        return out

    @staticmethod
    def load_model(model_file="best_model.pkl", meta_file="best_model_meta.json", device=None):
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

        # # ---- Load best weights ----
        # state_dict = torch.load(base_path + model_file, map_location=device)
        # model.load_state_dict(state_dict)
        with open(base_path + model_file, "rb") as f:
            state_dict = pickle.load(f)

        model.load_state_dict(state_dict)
        model.to(device)

        model.eval()
        return model, metadata


class OneHotLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 256, output_size: int = 16,
                 dropout_rate: float = 0.2, num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x_onehot: torch.FloatTensor, lengths: torch.LongTensor = None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x_onehot, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x_onehot.size(1))
        else:
            lstm_out, _ = self.lstm(x_onehot)

        out = self.dropout(lstm_out)
        logits = self.classifier(out)
        return logits
