from allennlp.common.params import Params
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BasicIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.modules import Embedding
from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate

from django.conf import settings
from django.core.files.storage import FileSystemStorage

import os

from .models import Language, ProbingTask

from .nn.dataset_readers.linspector_dataset_reader import LinspectorDatasetReader
from .nn.models.linspector_linear import LinspectorLinear
from .nn.models.multilayer_perceptron import MultilayerPerceptron

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Dict, Iterable, Iterator, List, Tuple

class Linspector():

    def __init__(self, language: Language, probing_tasks: List[ProbingTask], model: Model):
        self.language = language
        self.probing_tasks = probing_tasks
        self.archive = load_archive(model.model.path)
        self.model = self.archive.model

    def get_layers(self) -> List:
        layers = list()
        if isinstance(self.model, SimpleTagger):
            encoder = self.model.encoder._module
            modules_as_string = [str(module) for module in encoder.modules()]
            layers = [(idx, module[:module.index('(')]) for idx, module in enumerate(modules_as_string)]
        else:
            raise NotImplementedError
        return layers

    def _get_intrinsic_data(self, probing_task: ProbingTask, reader: DatasetReader) -> Tuple[Iterable[Instance], Iterable[Instance], Iterable[Instance]]:
        base_path = os.path.join(settings.MEDIA_ROOT, 'intrinsic_data', probing_task.to_camel_case(), self.language.code)
        # Read intrinsic vocab
        train = reader.read(os.path.join(base_path, 'train.txt'))
        dev = reader.read(os.path.join(base_path, 'dev.txt'))
        test = reader.read(os.path.join(base_path, 'test.txt'))
        return train, dev, test

    def _get_embeddings_from_model(self, probing_task: ProbingTask, layer: int) -> str:
        # Get intrinsic data for probing task
        # Set field_key to 'tokens' to work with SimpleTagger
        reader = LinspectorDatasetReader(ignore_labels=True, field_key='tokens')
        train, dev, test = self._get_intrinsic_data(probing_task, reader)
        # Select module
        module = list(self.model.encoder.modules())[layer]
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
                        self.model.forward_on_instance(instance)
                        # Write token and embedding to file
                        file.write('{} {}\n'.format(token, ' '.join(map(str, embedding.numpy().tolist()[0][0]))))
                        vocab.add(token)
        handle.remove()
        return name

    def _get_embedding_dim(self, embeddings_file: str) -> int:
        dim = 0
        file_system_storage = FileSystemStorage()
        with file_system_storage.open(embeddings_file, mode='r') as file:
            line = file.readline().rstrip()
            index = line.index(' ') + 1
            vector = np.array(line[index:].replace('  ', ' ').split(' '))
            dim = vector.size
        return dim

    def probe(self, layer: int) -> Dict[str, Any]:
        metrics = dict()
        for probing_task in self.probing_tasks:
            reader = LinspectorDatasetReader()
            train, dev, test = self._get_intrinsic_data(probing_task, reader)
            vocab = Vocabulary.from_instances(train + dev)
            embeddings_file = self._get_embeddings_from_model(probing_task, layer)
            file_system_storage = FileSystemStorage()
            params = Params({'embedding_dim': self._get_embedding_dim(embeddings_file), 'pretrained_file': file_system_storage.path(embeddings_file), 'trainable': False})
            word_embeddings = Embedding.from_params(vocab, params=params)
            model = LinspectorLinear(word_embeddings, vocab)
            # model = MultilayerPerceptron(word_embeddings, vocab, 2)
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
            metrics[probing_task.name] = evaluate(model, test, iterator, cuda_device, batch_weight_key='')
            file_system_storage.delete(embeddings_file)
        return metrics
