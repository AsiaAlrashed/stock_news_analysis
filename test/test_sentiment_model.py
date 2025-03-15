import unittest
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from models.sentiment_model import NewsClassifier


class TestSentimentModel(unittest.TestCase):
    """
    Unit tests for the sentiment analysis model
    """

    @classmethod
    def setUpClass(cls):
        """
        load the form once before all tests.
        """
        cls.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        cls.model = NewsClassifier(bert_model)
        cls.model.eval()  # Set the model to evaluation mode.
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model.to(cls.device)

    def test_model_loads_correctly(self):
        """
        Test the form loading without errors
        """
        self.assertIsNotNone(self.model)

    def test_prediction_shape(self):
        """
        Test that the model output has the correct dimensions.
        """
        sample_text = "Stock market is going up today!"
        encoding = self.tokenizer(
            sample_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        source = torch.tensor([[1.0]], dtype=torch.float32).to(
            self.device
        )  # Example of a news source

        with torch.no_grad():
            output = self.model(input_ids, attention_mask, source)

        self.assertEqual(
            output.shape[1], 3
        )  # We expect 3 categories (negative, neutral, positive)

    def test_batch_predictions(self):
        """
        Test that the model can handle a batch of data
        """
        sample_texts = [
            "Stock market is booming!",
            "Recession fears rise among investors.",
        ]
        encoding = self.tokenizer(
            sample_texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        sources = torch.tensor([[1.0], [0.0]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask, sources)

        self.assertEqual(
            output.shape, (2, 3)
        )  # Batch of 2 samples, output should be (2, 3)

    def test_model_inference_time(self):
        """
        Test that the inference time does not exceed a certain limit
        """
        import time

        sample_text = "The stock market has experienced a sharp decline."
        encoding = self.tokenizer(
            sample_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        source = torch.tensor([[0.5]], dtype=torch.float32).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            _ = self.model(input_ids, attention_mask, source)
        end_time = time.time()

        inference_time = end_time - start_time
        self.assertLess(
            inference_time, 0.5
        )  # The prediction time must be less than 0.5 seconds.


# if __name__ == "__main__":
#     unittest.main()
# python -m unittest discover tests
