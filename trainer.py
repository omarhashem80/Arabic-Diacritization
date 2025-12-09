import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import json
from sklearn.metrics import f1_score
from models import CharBiLSTM


class Trainer:
    def __init__(self, model, optimizer=None, scheduler=None, criterion=None,
                 train_loader=None, val_loader=None,
                 max_length=600, device=None, checkpoint_file="checkpoint.pth"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.max_length = max_length
        self.checkpoint_file = checkpoint_file
        self.best_f1 = -1
        self.best_model_state = None

    @staticmethod
    def load_checkpoint(checkpoint_file="checkpoint.pth",
                        model=None,
                        meta_file="best_model_meta.json",
                        device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_file, map_location=device)

        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if model is None:
            model = CharBiLSTM(
                vocab_size=metadata["vocab_size"],
                embedding_dim=metadata["embedding_size"],
                hidden_dim=metadata["hidden_size"],
                output_dim=metadata["output_classes"],
                dropout_rate=metadata["dropout_rate"],
                num_layers=metadata["num_layers"],
                max_length=metadata["max_sequence_length"]
            ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=metadata["learning_rate"])
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler and checkpoint['scheduler_state']:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']

        return model, optimizer, scheduler, start_epoch, best_f1

    def train_epoch(self):
        self.model.train()
        correct, total = 0, 0
        preds, trues = [], []

        for batch_seq, batch_labels in self.train_loader:
            batch_seq, batch_labels = batch_seq.to(self.device), batch_labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(batch_seq)
            flat_outputs = outputs.view(-1, outputs.shape[-1])
            flat_labels = batch_labels.view(-1)
            mask = (flat_labels != 15)

            loss = self.criterion(flat_outputs[mask], flat_labels[mask])
            loss.backward()
            self.optimizer.step()

            pred = flat_outputs.argmax(dim=1)
            correct += (pred[mask] == flat_labels[mask]).sum().item()
            total += mask.sum().item()
            preds.extend(pred[mask].cpu().tolist())
            trues.extend(flat_labels[mask].cpu().tolist())

        acc = correct / total
        f1 = f1_score(trues, preds, average='macro')
        return acc, f1, loss.item()

    def validate(self):
        self.model.eval()
        correct, total = 0, 0
        preds, trues = [], []

        with torch.inference_mode():
            for batch_seq, batch_labels in self.val_loader:
                batch_seq, batch_labels = batch_seq.to(self.device), batch_labels.to(self.device)
                outputs = self.model(batch_seq)

                pred = outputs.argmax(dim=2)
                flat_pred = pred.view(-1)
                flat_labels = batch_labels.view(-1)
                mask = (flat_labels != 15)

                correct += (flat_pred[mask] == flat_labels[mask]).sum().item()
                total += mask.sum().item()
                preds.extend(flat_pred[mask].cpu().tolist())
                trues.extend(flat_labels[mask].cpu().tolist())

        acc = correct / total
        f1 = f1_score(trues, preds, average='macro')
        return acc, f1

    def train(self, num_epochs=20):
        print("Training started...")
        for epoch in range(num_epochs):
            train_acc, train_f1, loss_val = self.train_epoch()
            val_acc, val_f1 = self.validate()

            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss_val:.5f} | "
                  f"Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.3f} | "
                  f"Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.3f}")

            # Save best model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_model_state = self.model.state_dict()
                print(f"New best model (Val F1={self.best_f1:.3f})")

            # Save checkpoint
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                'best_f1': self.best_f1,
                'epoch': epoch
            }, self.checkpoint_file)

        return self.best_model_state

    def test(self, test_loader):
        self.model.eval()
        correct, total = 0, 0

        with torch.inference_mode():
            for batch_seq, batch_labels in tqdm(test_loader, desc="Testing"):
                batch_seq, batch_labels = batch_seq.to(self.device), batch_labels.to(self.device)
                outputs = self.model(batch_seq)
                pred_labels = outputs.argmax(dim=2)

                mask = (batch_labels != 15)
                correct += ((pred_labels == batch_labels) & mask).sum().item()
                total += mask.sum().item()

        acc = correct / total if total > 0 else 0
        print(f"Test Accuracy: {acc*100:.3f}%")
        return acc
