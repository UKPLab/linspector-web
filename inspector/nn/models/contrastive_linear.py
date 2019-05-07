from allennlp.data import Vocabulary
from allennlp.modules import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy

from .linspector_linear import LinspectorLinear

from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

class ContrastiveLinear(LinspectorLinear):

    def __init__(self, word_embeddings: Embedding, vocab: Vocabulary) -> None:
        super().__init__(word_embeddings, vocab)
        # Set input dim times 2 for concatenated embeddings
        self.hidden2tag = nn.Linear(in_features=self.word_embeddings.get_output_dim() * 2, out_features=self.num_classes)

    @overrides
    def forward(self, first_token: Dict[str, torch.Tensor], second_token: Dict[str, torch.Tensor], label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        first_embedding = self.word_embeddings(first_token['tokens'])
        first_embedding = torch.squeeze(first_embedding, dim=1)
        second_embedding = self.word_embeddings(second_token['tokens'])
        second_embedding = torch.squeeze(second_embedding, dim=1)
        embedding = torch.cat((first_embedding, second_embedding), 1)
        # Masks from both tokens are identical
        mask = get_text_field_mask(first_token)
        logits = self.hidden2tag(embedding)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1)
        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}
        if label is not None:
            self.accuracy(logits, label, mask)
            output_dict['loss'] = sequence_cross_entropy_with_logits(logits, label, mask)
        return output_dict
