import torch
import re
import textwrap
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from cleaner import TextCleaner


def convert2idx(data, char_to_index, max_len=200, device='cpu'):
    sequences = [[char_to_index[ch] for ch in seq] for seq in data]
    sequences = [seq + [0] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(sequences, device=device)


class Predictor:
    def __init__(self, model, char_to_index, index_to_label, device=None):
        self.model = model
        self.char_to_index = char_to_index
        self.index_to_label = index_to_label
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.IGNORE_IDX = {0, 2, 8, 15, 16, 26, 40, 43}
    def predict_sentence(self, original_sentence, max_length=200, batch_size=256):
        # Clean & normalize
        clean = TextCleaner.clean_lines([original_sentence.strip()])[0]
        clean = re.sub(r'[\n\r\t]', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        tokenized_sentences = []

        # Split by dot
        parts = [p.strip() for p in clean.split('.') if p.strip()]

        # Wrap long strings without cutting words
        for part in parts:
            tokenized_sentences.extend(textwrap.wrap(part, max_length))

        seq_tensor = convert2idx(tokenized_sentences, self.char_to_index, max_length, self.device)
        loader = DataLoader(TensorDataset(seq_tensor, seq_tensor), batch_size=batch_size)

        predicted_labels = []

        # Run model
        self.model.eval()
        for batch_seq, batch_lbl in loader:
            outputs = self.model(batch_seq)
            batch_pred = outputs.argmax(dim=2)

            # Mask ignored chars
            mask = ~torch.isin(batch_seq, torch.tensor(list(self.IGNORE_IDX), device=self.device))
            predicted_labels.extend(batch_pred[mask].tolist())


        predicted_sentence = ""
        idx = 0

        for ch in original_sentence:
            predicted_sentence += ch

            if ch not in self.char_to_index:
                continue

            code = self.char_to_index[ch]
            if code in self.IGNORE_IDX:
                continue

            pred_class = self.index_to_label[predicted_labels[idx]]

            if pred_class == 0:
                idx += 1
                continue

            if isinstance(pred_class, tuple):
                predicted_sentence += chr(pred_class[0]) + chr(pred_class[1])
            else:
                predicted_sentence += chr(pred_class)

            idx += 1

        return predicted_sentence

    def predict_dataset(self, dataloader):
        predicted_labels = []
        ignore_indices = {0, 2, 8, 15, 16, 26, 40, 43}

        self.model.eval()
        with torch.inference_mode():
            for batch_seq, _ in dataloader:
                batch_seq = batch_seq.to(self.device)
                outputs = self.model(batch_seq)
                batch_pred = outputs.argmax(dim=2)

                mask = ~torch.isin(batch_seq, torch.tensor(list(ignore_indices), device=self.device))

                predicted_labels.extend(batch_pred[mask].tolist())

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
