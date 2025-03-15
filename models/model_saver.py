import torch


class ModelSaver:
    """
    Class to save and load the form.
    """

    @staticmethod
    def save_model(model, filepath="models/news_classifier.pth"):
        """
        Save the form to a file.
        """
        torch.save(model.state_dict(), filepath)
        print(f"The model has been saved at {filepath}")

    @staticmethod
    def load_model(
        model_class, bert_model, filepath="models/news_classifier.pth", device="cpu"
    ):
        """
        Load the saved form.
        """
        model = model_class(bert_model).to(device)
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.eval()
        print(f"Model loaded from {filepath}")
        return model
