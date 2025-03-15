import torch
import torch.nn as nn
from transformers import DistilBertModel


class NewsClassifier(nn.Module):
    """
    Text analysis model using DistilBERT and GRU.
    """

    def __init__(self, bert_model, hidden_size=128, num_classes=3):
        super(NewsClassifier, self).__init__()
        self.bert = bert_model
        self.bert.requires_grad_(True)  # Enable Fine-Tuning

        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(768 + hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, source):
        bert_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        source = source.unsqueeze(1)  # Adjust dimensions for GRU
        _, gru_output = self.gru(source)
        gru_output = gru_output.squeeze(0)

        combined = torch.cat((bert_output, gru_output), dim=1)
        output = self.fc(combined)
        return output
