from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinspectorLinear(Model):

    def __init__(self, word_embeddings, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.hidden2tag = nn.Linear(in_features=self.word_embeddings.get_output_dim(), out_features=self.num_classes)
        self.accuracy = CategoricalAccuracy()

    def forward(self, token, label = None):
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

    def decode(self, output_dict):
        predictions = output_dict['class_probabilities']
        predictions = predictions.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace='labels') for x in argmax_indices]
        output_dict['labels'] = labels
        return output_dict

    def get_metrics(self, reset = False):
        return {'accuracy': self.accuracy.get_metric(reset)}
