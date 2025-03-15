import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from models.sentiment_model import NewsClassifier
from src.data_loader import DataLoad
from src.data_preparation import DataPreparer
from evaluate_model import ModelEvaluator
from model_saver import ModelSaver
from sklearn.model_selection import train_test_split


class NewsDataset(Dataset):
    def __init__(self, headlines, sources, labels, tokenizer, max_length=128):
        self.headlines = headlines
        self.sources = sources
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.headlines[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        source = torch.tensor([self.sources[idx]], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return input_ids, attention_mask, source, label


class ModelTrainer:
    """
    Class for training a news sentiment analysis model.
    """

    def __init__(self, data_path, batch_size=16, num_epochs=10):
        self.data_loader = DataLoad(data_path)
        self.data_preparer = DataPreparer()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NewsClassifier(self.bert_model).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self._prepare_data()

    def _prepare_data(self):
        """
        Loading and preparing data and creating a DataLoader
        """
        df = self.data_loader.load_data()
        df = self.data_preparer.prepare_data(df)

        headlines = df["clean_headline"].tolist()
        sources = df["source_encoded"].values
        labels = df["final_label_encoded"].values

        # Splitting data into training and testing
        (
            train_texts,
            test_texts,
            train_sources,
            test_sources,
            train_labels,
            test_labels,
        ) = train_test_split(
            headlines, sources, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Create `Dataset` objects
        train_dataset = NewsDataset(
            train_texts, train_sources, train_labels, self.tokenizer
        )
        test_dataset = NewsDataset(
            test_texts, test_sources, test_labels, self.tokenizer
        )

        # Create DataLoader
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

    def train(self):
        """
        Model training
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for input_ids, attention_mask, source, labels in self.train_loader:
                input_ids, attention_mask, source, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    source.to(self.device),
                    labels.to(self.device),
                )

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, source)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(self.train_loader)}")

    def get_model(self):
        """
        Trained Model Return
        """
        return self.model


# Running training
trainer = ModelTrainer("data/processed_news.csv")
trainer.train()

# Extract the trained model
model = trainer.get_model()

# Device identification
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Evaluation
evaluator = ModelEvaluator(model, device)
accuracy = evaluator.evaluate(trainer.test_loader)

# Save form
ModelSaver.save_model(model, filepath="models/news_classifier.pth")

print(f"Model accuracy: {accuracy:.4f}")
print("The model has been saved successfully!")
