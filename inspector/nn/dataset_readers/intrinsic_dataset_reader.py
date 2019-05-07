from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

import os

from overrides import overrides

from typing import Iterator, List

class IntrinsicDatasetReader(DatasetReader):

    def __init__(self, field_key: str, contrastive: bool = False) -> None:
        super().__init__(lazy=False)
        self.field_key = field_key
        self.contrastive = contrastive
        self.token_indexers = {'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}

    @overrides
    def text_to_instance(self, token: List[Token]) -> Instance:
        token_field = TextField(token, self.token_indexers)
        fields = {self.field_key: token_field}
        return Instance(fields)

    @overrides
    def _read(self, base_path: str) -> Iterator[Instance]:
        # Use set to filter duplicates
        vocab = set()
        for source in ['train.txt', 'dev.txt', 'test.txt']:
            with open(os.path.join(base_path, source)) as data:
                for line in data:
                    split = line.strip().split('\t')
                    # Lowercase tokens
                    vocab.add(split[0].lower())
                    if self.contrastive:
                        vocab.add(split[1].lower())
        for token in sorted(vocab):
            yield self.text_to_instance([Token(token)])
