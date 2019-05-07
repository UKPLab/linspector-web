from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from overrides import overrides

from typing import Iterator, List

class LinspectorDatasetReader(DatasetReader):

    def __init__(self) -> None:
        super().__init__(lazy=False)
        self.token_indexers = {'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}

    @overrides
    def text_to_instance(self, token: List[Token], label: str = None) -> Instance:
        token_field = TextField(token, self.token_indexers)
        fields = {'token': token_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as file:
            for line in file:
                split = line.strip().split('\t')
                token = Token(split[0])
                if len(split) > 1:
                    label = split[1]
                else:
                    label = None
                yield self.text_to_instance([token], label)
