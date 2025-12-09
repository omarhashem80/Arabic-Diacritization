import torch
import re
import textwrap
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Predictor:
    def __init__(self, model, char_to_index, index_to_label, device=None):
        self.model = model
        self.char_to_index = char_to_index
        self.index_to_label = index_to_label
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_sentence(self, sentence, max_length=200, batch_size=256):
        # Preprocess and tokenize the sentence
        cleaned = re.sub(r'\s+', ' ', sentence.strip())
        tokenized = textwrap.wrap(cleaned, max_length)

        sequences = [
            [self.char_to_index.get(c, 0) for c in s] + [0] * (max_length - len(s))
            for s in tokenized
        ]
        sequences_tensor = torch.tensor(sequences).to(self.device)

        dataset = TensorDataset(sequences_tensor, sequences_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        predicted_labels = []

        self.model.eval()
        with torch.inference_mode():
            for batch_seq, _ in dataloader:
                outputs = self.model(batch_seq)
                batch_pred = outputs.argmax(dim=2)
                mask = batch_seq != 0
                predicted_labels.extend(batch_pred[mask].tolist())

        # Reconstruct sentence with diacritics
        reconstructed = ""
        idx = 0
        for char in sentence:
            reconstructed += char
            if char not in self.char_to_index or self.char_to_index[char] in [0, 2, 8, 15, 16, 26, 40, 43]:
                continue
            predicted_class = self.index_to_label[predicted_labels[idx]]
            if isinstance(predicted_class, tuple):
                reconstructed += chr(predicted_class[0]) + chr(predicted_class[1])
            elif predicted_class != 0:
                reconstructed += chr(predicted_class)
            idx += 1

        return reconstructed

    def predict_dataset(self, dataloader):
        # Predict labels for a full dataset and save to submission.csv
        predicted_labels = []

        self.model.eval()
        with torch.inference_mode():
            for batch_seq, _ in dataloader:
                batch_seq = batch_seq.to(self.device)
                outputs = self.model(batch_seq)
                batch_pred = outputs.argmax(dim=2)

                mask = (batch_seq != 0) & (batch_seq != 2) & (batch_seq != 8) & \
                       (batch_seq != 15) & (batch_seq != 16) & (batch_seq != 26) & \
                       (batch_seq != 40) & (batch_seq != 43)

                predicted_labels.extend(batch_pred[mask].tolist())

        # Save predictions
        with open('submission.csv', 'w', encoding='utf-8') as f:
            f.write('ID,label\n')
            for i, label in enumerate(predicted_labels):
                f.write(f"{i},{label}\n")

        return predicted_labels

    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0

        with torch.inference_mode():
            for seq, labels in tqdm(test_loader, desc="Evaluating"):
                seq = seq.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(seq)
                pred = outputs.argmax(dim=2)

                flat_pred = pred.view(-1)
                flat_labels = labels.view(-1)

                # mask out PAD = 15
                mask = (flat_labels != 15)

                correct += (flat_pred[mask] == flat_labels[mask]).sum().item()
                total += mask.sum().item()

        return correct / total if total > 0 else 0
