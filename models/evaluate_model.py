import torch


class ModelEvaluator:
    """
    Class to evaluate the accuracy of the model on the test set.
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, test_loader):
        """
        Model evaluation and accuracy calculation
        """
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for input_ids, attention_mask, source, labels in test_loader:
                input_ids, attention_mask, source, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    source.to(self.device),
                    labels.to(self.device),
                )

                outputs = self.model(input_ids, attention_mask, source)
                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Model Accuracy: {accuracy:.4f}")
        return accuracy
