import torch
from torch.utils.data import DataLoader, TensorDataset


class DatasetBuilder:
    def __init__(self, preprocessor, char_to_index, label_map, max_length=600, device='cpu'):
        self.preprocessor = preprocessor
        self.char_to_index = char_to_index
        self.label_map = label_map
        self.max_length = max_length
        self.device = device

    def encode_chars(self, sequences):
        indexed = [[self.char_to_index[char] for char in seq] for seq in sequences]
        padded = [seq + [0] * (self.max_length - len(seq)) for seq in indexed]
        return torch.tensor(padded).to(self.device)

    def encode_labels(self, sequences_with_diacritics):
        all_labels = []

        for sentence in sequences_with_diacritics:
            labels = []
            i = 0
            length = len(sentence)

            while i < length:
                ch = ord(sentence[i])

                # Case 1: base letter (not a diacritic)
                if ch not in self.label_map:

                    # Check next character
                    if i + 1 < length:
                        d1 = ord(sentence[i + 1])

                        # Case 1a: shadda + another diacritic
                        if d1 == 1617 and i + 2 < length:
                            d2 = ord(sentence[i + 2])
                            if d2 in self.label_map:
                                labels.append(self.label_map[(1617, d2)])
                                i += 3
                                continue

                        # Case 1b: shadda alone
                        if d1 == 1617:
                            labels.append(self.label_map[1617])
                            i += 2
                            continue

                        # Case 1c: single diacritic
                        if d1 in self.label_map:
                            labels.append(self.label_map[d1])
                            i += 2
                            continue

                    # Case 1d: no diacritic
                    labels.append(14)

                i += 1

            # pad
            if len(labels) < self.max_length:
                labels.extend([15] * (self.max_length - len(labels)))

            all_labels.append(labels)

        return torch.tensor(all_labels, device=self.device)

    def create_dataloader(self, data_type, batch_size=256, with_labels=True):
        self.preprocessor.preprocess_file(data_type)
        sequences, sequences_with_diacritics = self.preprocessor.tokenize_file(data_type)

        data_tensor = self.encode_chars(sequences)

        if with_labels:
            labels_tensor = self.encode_labels(sequences_with_diacritics)
            dataset = TensorDataset(data_tensor, labels_tensor)
        else:
            dataset = TensorDataset(data_tensor)

        return DataLoader(dataset, batch_size=batch_size)

