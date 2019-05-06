from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy

from overrides import overrides

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

class LinspectorLinear(Model):

    def __init__(self, word_embeddings: Embedding, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.hidden2tag = nn.Linear(in_features=self.word_embeddings.get_output_dim(), out_features=self.num_classes)
        self.accuracy = CategoricalAccuracy()

    @overrides
    def forward(self, token: Dict[str, torch.Tensor], label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        embedding = self.word_embeddings(token['tokens'])
        embedding = torch.squeeze(embedding, dim=1)
        mask = get_text_field_mask(token)
        logits = self.hidden2tag(embedding)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1)
        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}
        if label is not None:
            self.accuracy(logits, label, mask)
            output_dict['loss'] = sequence_cross_entropy_with_logits(logits, label, mask)
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities']
        predictions = predictions.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace='labels') for x in argmax_indices]
        output_dict['labels'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}
