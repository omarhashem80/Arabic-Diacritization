import torch
from torch.optim.lr_scheduler import StepLR
from cleaner import TextCleaner
from preprocessor import TextPreprocessor
from dataset_builder import DatasetBuilder
from models import CharBiLSTM
from trainer import Trainer
from predictor import Predictor
from tqdm import tqdm
import gradio as gr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHAR_TO_INDEX = {
    "د": 1,
    "؟": 2,
    "آ": 3,
    "إ": 4,
    "ؤ": 5,
    "ط": 6,
    "م": 7,
    "،": 8,
    "ة": 9,
    "ت": 10,
    "ر": 11,
    "ئ": 12,
    "ا": 13,
    "ض": 14,
    "!": 15,
    " ": 16,
    "ك": 17,
    "غ": 18,
    "س": 19,
    "ص": 20,
    "أ": 21,
    "ل": 22,
    "ف": 23,
    "ظ": 24,
    "ج": 25,
    "؛": 26,
    "ن": 27,
    "ع": 28,
    "ب": 29,
    "ث": 30,
    "ه": 31,
    "خ": 32,
    "ى": 33,
    "ء": 34,
    "ز": 35,
    "ق": 36,
    "ي": 37,
    "ش": 38,
    "ح": 39,
    ":": 40,
    "ذ": 41,
    "و": 42,
    ".": 43,
}
INDEX_TO_CHAR = {v: k for k, v in CHAR_TO_INDEX.items()}

LABELS = {
    1614: 0,
    1611: 1,
    1615: 2,
    1612: 3,
    1616: 4,
    1613: 5,
    1618: 6,
    1617: 7,
    (1617, 1614): 8,
    (1617, 1611): 9,
    (1617, 1615): 10,
    (1617, 1612): 11,
    (1617, 1616): 12,
    (1617, 1613): 13,
    0: 14,
    15: 15,
}
INDEX_TO_LABEL = {v: k for k, v in LABELS.items()}

MAX_LENGTH = 600
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 256
READ_PATH = "data/"
WRITE_PATH = "cleaned_outputs/"


def test_last_char_text(
    model,
    data_loader,
    max_len=600,
    batch_size=256,
    char_to_index=CHAR_TO_INDEX,
    index_to_label=INDEX_TO_LABEL,
    labels=LABELS,
    index_to_char=INDEX_TO_CHAR,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model.eval()

    total_last_char = 0
    correct_last_char = 0
    words_text = []

    with torch.inference_mode():
        for batch_sequences, batch_labels in tqdm(data_loader, desc="Last Char Test"):
            outputs = model(batch_sequences)  # batch_size * seq_length * output_size
            predicted_labels = outputs.argmax(dim=2)

            batch_size_seq = batch_sequences.shape[0]

            for i in range(batch_size_seq):
                seq = batch_sequences[i]
                true_labels = batch_labels[i]
                pred_labels = predicted_labels[i]

                word = ""
                last_char_true = None
                last_char_pred = None

                for idx, c in enumerate(seq):
                    if c in [0, 2, 8, 15, 16, 26, 40, 43]:
                        if word:
                            last_char_true_val = (
                                last_char_true if last_char_true is not None else 0
                            )
                            last_char_pred_val = (
                                last_char_pred if last_char_pred is not None else 0
                            )
                            words_text.append(
                                f"{word}:{last_char_true_val}->{last_char_pred_val}"
                            )
                            if last_char_true_val == last_char_pred_val:
                                correct_last_char += 1
                            total_last_char += 1

                            word = ""
                            last_char_true = None
                            last_char_pred = None
                        continue

                    word += index_to_char[int(c)]
                    last_char_true = index_to_label[int(true_labels[idx])]
                    last_char_pred = index_to_label[int(pred_labels[idx])]

                # catch last word
                if word:
                    last_char_true_val = (
                        last_char_true if last_char_true is not None else 0
                    )
                    last_char_pred_val = (
                        last_char_pred if last_char_pred is not None else 0
                    )
                    words_text.append(
                        f"{word}:{last_char_true_val}->{last_char_pred_val}"
                    )
                    if last_char_true_val == last_char_pred_val:
                        correct_last_char += 1
                    total_last_char += 1

    accuracy = correct_last_char / total_last_char if total_last_char > 0 else 0
    print(f"Last Character Accuracy: {accuracy*100:.3f}%")

    return accuracy


if __name__ == "__main__":
    # Initialize preprocessing classes
    cleaner = TextCleaner()
    preprocessor = TextPreprocessor(
        cleaner, input_path=READ_PATH, output_path=WRITE_PATH
    )
    dataset_builder = DatasetBuilder(
        preprocessor,
        char_to_index=CHAR_TO_INDEX,
        label_map=LABELS,
        max_length=MAX_LENGTH,
        device=DEVICE,
    )

    # Prepare dataloaders
    # train_loader = dataset_builder.create_dataloader(
    #     data_type="train", batch_size=TRAIN_BATCH_SIZE
    # )
    # val_loader = dataset_builder.create_dataloader(
    #     data_type="val", batch_size=VAL_BATCH_SIZE
    # )
    # test_loader = dataset_builder.create_dataloader(
    #     data_type="test", batch_size=VAL_BATCH_SIZE, with_labels=False
    # )

    # Initialize model
    vocab_size = len(CHAR_TO_INDEX) + 1
    embedding_dim = 300
    hidden_dim = 256
    output_dim = len(LABELS)
    dropout_rate = 0.2
    num_layers = 5

    model = CharBiLSTM(
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        dropout_rate,
        num_layers,
        max_length=MAX_LENGTH,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the model
    # criterion = torch.nn.CrossEntropyLoss()
    # trainer = Trainer(
    #     model=model,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     criterion=criterion,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     device=DEVICE,
    #     checkpoint_file="checkpoint.pth",
    # )

    # trainer.train(num_epochs=20)

    # Load best model for prediction
    model, meta = CharBiLSTM.load_model_pth()
    predictor = Predictor(model, CHAR_TO_INDEX, INDEX_TO_LABEL, device=DEVICE)

    def diacritize_text(text):
        if not text or not text.strip():
            return ""

        try:
            predicted_sentence = predictor.predict_sentence(text, max_length=MAX_LENGTH)
            return predicted_sentence
        except Exception as e:
            return f"Error during prediction: {str(e)}"

    iface = gr.Interface(
        fn=diacritize_text,
        inputs=gr.Textbox(
            lines=2,
            placeholder="Enter Arabic text here...",
            label="Input Sentence",
            rtl=True,
        ),
        outputs=gr.Textbox(label="Diacritized Sentence", rtl=True),
        title="Arabic Diacritization",
        description="Enter an Arabic sentence to predict its diacritics.",
    )
    iface.launch()
    # Predict on test dataset
    # predictor.predict_dataset(test_loader)

    # Predict a single sentence
    # test_sentence = ''
    # predicted_sentence = predictor.predict_sentence(test_sentence, max_length=MAX_LENGTH, batch_size=VAL_BATCH_SIZE)
    # print("Original sentence:", test_sentence)
    # print("Predicted sentence:", predicted_sentence)

    # Evaluate on validation set
    # predictor.evaluate(val_loader)
