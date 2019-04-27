from allennlp.common.params import Params
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.modules import Embedding, FeedForward, TextFieldEmbedder
from allennlp.nn.activations import Activation
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate

from django.conf import settings
from django.core.files.storage import FileSystemStorage

import os

from overrides import overrides

from .models import Language, ProbingTask

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

def load_model(path: str) -> nn.Module:
    archive = load_archive(path)
    return archive.model

def get_modules(model: nn.Module) -> List:
    modules = list()
    if isinstance(model, SimpleTagger):
        encoder = model.encoder._module
        modules_as_string = [str(module) for module in encoder.modules()]
        modules = [(idx, module[:module.index('(')]) for idx, module in enumerate(modules_as_string)]
    else:
        raise NotImplementedError
    return modules

def get_intrinsic_data(probing_task: ProbingTask, language: Language, reader: DatasetReader) -> Tuple[Iterable[Instance], Iterable[Instance], Iterable[Instance]]:
    base_path = os.path.join(settings.MEDIA_ROOT, 'intrinsic_data', probing_task.to_camel_case(), language.code)
    # Read intrinsic vocab
    train = reader.read(os.path.join(base_path, 'train.txt'))
    dev = reader.read(os.path.join(base_path, 'dev.txt'))
    test = reader.read(os.path.join(base_path, 'test.txt'))
    return train, dev, test

def get_embeddings_from_model(probing_task: ProbingTask, language: Language, model: Model, layer: int) -> str:
    # Get intrinsic data for probing task
    # Set field_key to 'tokens' to work with SimpleTagger
    reader = LinspectorDatasetReader(ignore_labels=True, field_key='tokens')
    train, dev, test = get_intrinsic_data(probing_task, language, reader)
    # Select module
    module = list(model.encoder.modules())[layer]
    # Get embeddings for vocab
    embedding = torch.zeros((1, 1, module.get_input_dim()))
    def hook(module, input, output):
        # input[0] contains a torch.nn.utils.rnn.PackedSequence which also has a batch_sizes property
        embedding.copy_(input[0].data)
    handle = module.register_forward_hook(hook)
    file_system_storage = FileSystemStorage()
    name = file_system_storage.get_available_name('probe.vec')
    vocab = set()
    with file_system_storage.open(name, mode='w') as file:
        with torch.no_grad():
            for instance in train + dev + test:
                # Lowercase tokens
                token = str(instance['tokens'][0]).lower()
                # Filter duplicates
                if token not in vocab:
                    model.forward_on_instance(instance)
                    # Write token and embedding to file
                    file.write('{} {}\n'.format(token, ' '.join(map(str, embedding.numpy().tolist()[0][0]))))
                    vocab.add(token)
    handle.remove()
    return name

def get_embedding_dim(embeddings_file: str) -> int:
    dim = 0
    file_system_storage = FileSystemStorage()
    with file_system_storage.open(embeddings_file, mode='r') as file:
        line = file.readline().rstrip()
        index = line.index(' ') + 1
        vector = np.array(line[index:].replace('  ', ' ').split(' '))
        dim = vector.size
    return dim

def probe(probing_task: ProbingTask, language: Language, embeddings_file: str) -> Dict[str, Any]:
    reader = LinspectorDatasetReader()
    train, dev, test = get_intrinsic_data(probing_task, language, reader)
    vocab = Vocabulary.from_instances(train + dev)
    file_system_storage = FileSystemStorage()
    params = Params({'embedding_dim': get_embedding_dim(embeddings_file), 'pretrained_file': file_system_storage.path(embeddings_file), 'trainable': False})
    word_embeddings = Embedding.from_params(vocab, params=params)
    model = LinspectorLinear(word_embeddings, vocab)
    # model = LinspectorMultilayerPerceptron(word_embeddings, vocab, 2)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    optimizer = optim.Adam(model.parameters())
    iterator = BasicIterator(batch_size=16)
    iterator.index_with(vocab)
    trainer = Trainer(model=model, optimizer=optimizer, iterator=iterator, train_dataset=train, validation_dataset=dev, patience=5, num_epochs=20, cuda_device=cuda_device)
    trainer.train()
    metrics = evaluate(model, test, iterator, cuda_device, batch_weight_key='')
    return metrics

class LinspectorDatasetReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, ignore_labels: bool = False, field_key: str = 'token') -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}
        self._ignore_labels = ignore_labels
        self._field_key = field_key

    def text_to_instance(self, token: List[Token], label: str = None) -> Instance:
        token_field = TextField(token, self.token_indexers)
        fields = {self._field_key: token_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

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

class LinspectorLinear(Model):

    def __init__(self, word_embeddings: Embedding, vocab: Vocabulary) -> None:
        super(LinspectorLinear, self).__init__(vocab)
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

class LinspectorMultilayerPerceptron(LinspectorLinear):

    def __init__(self, word_embeddings: Embedding, vocab: Vocabulary, num_layers: int = 2) -> None:
        super(LinspectorMultilayerPerceptron, self).__init__(word_embeddings, vocab)
        self.hidden2tag = FeedForward(input_dim=self.word_embeddings.get_output_dim(), num_layers=num_layers, hidden_dims=[self.word_embeddings.get_output_dim(), self.num_classes], activations=[Activation.by_name('relu')(), Activation.by_name('linear')()], dropout=[0.5, 0.0])
