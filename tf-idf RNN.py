# Import necessary libraries
import torch
import torch.nn as nn
import pickle
import numpy as np
import re
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn import ReLU
from tqdm import tqdm
from torch.utils.data import TensorDataset
import sys
import os
from cleaner import TextCleaner
from preprocessor import TextPreprocessor
from main import LABELS, INDEX_TO_LABEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, hidden = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output


class RNN():
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.training_data = []
        self.validation_data = []
        self.training_labels = []
        self.validation_labels = []
        self.vectorizer = None
        self.model = None
        self.cleaner = TextCleaner()
        self.model_path = 'RNN_TF-IDF.pth'
        self.vectorizer_path = 'tfidf_vectorizer.pkl'

    def preprocess_and_clean_data(self):
        cleaner = TextCleaner()
        preprocessor = TextPreprocessor(
            cleaner=cleaner,
            input_path=self.data_path,
            output_path=self.data_path,
            max_length=600,
            with_labels=True
        )
        
        preprocessor.preprocess_file('train')
        train_no_diacritics, train_with_diacritics = preprocessor.tokenize_file('train')
        
        preprocessor.preprocess_file('val')
        val_no_diacritics, val_with_diacritics = preprocessor.tokenize_file('val')
        
        self.training_data = train_no_diacritics
        self.validation_data = val_no_diacritics
        
        self.training_labels = self.extract_labels_from_text(train_with_diacritics)
        self.validation_labels = self.extract_labels_from_text(val_with_diacritics)
        
        print(f"Training samples: {len(self.training_data)}")
        print(f"Validation samples: {len(self.validation_data)}")
        
    def extract_labels_from_text(self, diacritized_texts):
        labels = []
        unicode_to_label = {}
        for unicode_val, label_idx in LABELS.items():
            if isinstance(unicode_val, int) and unicode_val != 0 and unicode_val != 15:
                unicode_to_label[unicode_val] = label_idx
        
        for text in diacritized_texts:
            label = 14
            for char in text:
                char_code = ord(char)
                if char_code in unicode_to_label:
                    label = unicode_to_label[char_code]
                    break
            labels.append(label)
        
        return labels

    def train(self, vectorizer):
        self.vectorizer = vectorizer
        
        X_train_tfidf = self.vectorizer.fit_transform(self.training_data)
        X_train = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
        y_train = torch.tensor(self.training_labels, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        input_size = X_train.shape[1]
        hidden_size = 128
        output_size = len(LABELS)
        num_epochs = 15

        model = RNNModel(input_size, hidden_size, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                batch_features = batch_features.unsqueeze(1)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        torch.save(model.state_dict(), 'RNN_TF-IDF.pth')
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        return model

    def validate(self, model=None):
        if model is None:
            input_size = len(self.vectorizer.get_feature_names_out())
            hidden_size = 128
            output_size = 15
            model = RNNModel(input_size, hidden_size, output_size).to(device)
            model.load_state_dict(torch.load('RNN_TF-IDF.pth'))
        
        model.eval()
        X_val_tfidf = self.vectorizer.transform(self.validation_data)
        X_val = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float32).to(device)
        y_val = torch.tensor(self.validation_labels, dtype=torch.long).to(device)

        with torch.no_grad():
            X_val = X_val.unsqueeze(1)
            outputs = model(X_val)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_val).sum().item() / len(y_val)
        
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found!")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file '{self.vectorizer_path}' not found!")
        
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        input_size = len(self.vectorizer.get_feature_names_out())
        hidden_size = 128
        output_size = len(LABELS)
        
        self.model = RNNModel(input_size, hidden_size, output_size).to(device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model.eval()
        print("Model and vectorizer loaded successfully!")
    
    def clean_sentence(self, sentence):
        lines = [sentence]
        lines = self.cleaner.clean_lines(lines)
        lines = self.cleaner.remove_diacritics(lines)
        return lines[0] if lines else ""
    
    def predict(self, sentence):
        if self.model is None or self.vectorizer is None:
            print("Loading model...")
            self.load_model()
        
        cleaned_sentence = self.clean_sentence(sentence)
        
        if not cleaned_sentence.strip():
            print("Empty sentence after cleaning!")
            return ""
        
        predicted_sentence = []
        chars = list(cleaned_sentence)
        
        for i, char in enumerate(chars):
            if char.strip():
                context_window = cleaned_sentence[max(0, i-10):min(len(cleaned_sentence), i+10)]
                
                X_tfidf = self.vectorizer.transform([context_window])
                X = torch.tensor(X_tfidf.toarray(), dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    X = X.unsqueeze(1)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    predicted_label = predicted.item()
                
                label_unicode = INDEX_TO_LABEL.get(predicted_label, 0)
                
                predicted_sentence.append(char)
                
                if isinstance(label_unicode, tuple):
                    diacritic = ''.join([chr(code) for code in label_unicode])
                    predicted_sentence.append(diacritic)
                elif label_unicode != 0 and label_unicode != 15:
                    diacritic = chr(label_unicode)
                    predicted_sentence.append(diacritic)
            else:
                predicted_sentence.append(char)
        
        predicted_sentence = ''.join(predicted_sentence)
        
        print(f"\nOriginal: {sentence}")
        print(f"Cleaned: {cleaned_sentence}")
        print(f"Predicted: {predicted_sentence}")
        
        return predicted_sentence


if __name__ == "__main__":
    rnn = RNN(data_path='./data/')
    
    if os.path.exists('RNN_TF-IDF.pth') and os.path.exists('tfidf_vectorizer.pkl'):
        print("Model files found. Loading existing model...")
        rnn.load_model()
        
        test_sentence = "قَالَ أَبُو زَيْدٍ أَهْلُ تِهَامَةَ يُؤَنِّثُونَ الْعَضُدَ وَبَنُو تَمِيمٍ يُذَكِّرُونَ ، وَالْجَمْعُ أَعْضُدٌ وَأَعْضَادٌ مِثْلُ أَفْلُسٍ وَأَقْفَالٍ"
        rnn.predict(test_sentence)
    else:
        print("\nArabic Diacritization RNN Training Pipeline\n")
        
        rnn.preprocess_and_clean_data()
        
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 3),
            max_features=5000,
            min_df=2
        )
        
        trained_model = rnn.train(vectorizer=vectorizer)
        rnn.validate(model=trained_model)
        
        print("\nTraining Complete!")
        
        rnn.model = trained_model
        test_sentence = "قَالَ أَبُو زَيْدٍ أَهْلُ تِهَامَةَ يُؤَنِّثُونَ الْعَضُدَ وَبَنُو تَمِيمٍ يُذَكِّرُونَ ، وَالْجَمْعُ أَعْضُدٌ وَأَعْضَادٌ مِثْلُ أَفْلُسٍ وَأَقْفَالٍ"
        rnn.predict(test_sentence)