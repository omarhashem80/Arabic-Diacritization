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
        # Convert characters to indices and pad sequences
        indexed = [[self.char_to_index[char] for char in seq] for seq in sequences]
        padded = [seq + [0] * (self.max_length - len(seq)) for seq in indexed]
        return torch.tensor(padded).to(self.device)

    def encode_labels(self, sequences_with_diacritics):
        # Convert diacritics to label indices and pad sequences
        all_labels = []
        for sentence in sequences_with_diacritics:
            sentence_labels = []
            i = 0
            while i < len(sentence):
                char_code = ord(sentence[i])
                if char_code not in self.label_map:
                    if (i + 1 < len(sentence)) and ord(sentence[i + 1]) in self.label_map:
                        diacritic = ord(sentence[i + 1])
                        if diacritic == 1617 and (i + 2 < len(sentence)) and ord(sentence[i + 2]) in self.label_map:
                            sentence_labels.append(self.label_map[(1617, ord(sentence[i + 2]))])
                            i += 3
                            continue
                        elif diacritic == 1617:
                            sentence_labels.append(self.label_map[1617])
                            i += 2
                            continue
                        else:
                            sentence_labels.append(self.label_map[diacritic])
                            i += 2
                            continue
                    else:
                        sentence_labels.append(14)  # No diacritic
                i += 1
            all_labels.append(sentence_labels)

        # Pad label sequences
        padded_labels = [seq + [15] * (self.max_length - len(seq)) for seq in all_labels]
        return torch.tensor(padded_labels).to(self.device)

    def create_dataloader(self, data_type, batch_size=256, with_labels=True):
        # Preprocess and tokenize
        self.preprocessor.preprocess_file(data_type)
        sequences, sequences_with_diacritics = self.preprocessor.tokenize_file(data_type)

        # Encode characters
        data_tensor = self.encode_chars(sequences)

        # Encode labels
        if with_labels:
            labels_tensor = self.encode_labels(sequences_with_diacritics)
        else:
            labels_tensor = torch.tensor([[15] * self.max_length] * len(data_tensor)).to(self.device)

        dataset = TensorDataset(data_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=batch_size)
