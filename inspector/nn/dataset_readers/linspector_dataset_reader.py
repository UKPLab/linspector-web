from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from overrides import overrides

from typing import Dict, Iterator, List

class LinspectorDatasetReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, ignore_labels: bool = False, field_key: str = 'token') -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}
        self._ignore_labels = ignore_labels
        self._field_key = field_key

    @overrides
    def text_to_instance(self, token: List[Token], label: str = None) -> Instance:
        token_field = TextField(token, self.token_indexers)
        fields = {self._field_key: token_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as file:
            for line in file:
                split = line.strip().split('\t')
                token = Token(split[0])
                if len(split) > 1 and not self._ignore_labels:
                    label = split[1]
                else:
                    label = None
                yield self.text_to_instance([token], label)
