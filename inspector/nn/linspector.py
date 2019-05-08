from abc import ABC

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.modules import Embedding
from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate

from django.conf import settings

import os

from inspector.models import Language, ProbingTask
from .dataset_readers.contrastive_dataset_reader import ContrastiveDatasetReader
from .dataset_readers.intrinsic_dataset_reader import IntrinsicDatasetReader
from .dataset_readers.linspector_dataset_reader import LinspectorDatasetReader
from .models.linspector_linear import LinspectorLinear
from .models.contrastive_linear import ContrastiveLinear

import numpy as np

import shutil

from tempfile import NamedTemporaryFile

import torch
import torch.nn as nn
import torch.optim as optim

class Linspector(ABC):

    def __init__(self, language, probing_tasks):
        self.language = language
        self.probing_tasks = probing_tasks

    def _get_intrinsic_data(self, probing_task):
        base_path = os.path.join(settings.MEDIA_ROOT, 'intrinsic_data', probing_task.to_camel_case(), self.language.code)
        if probing_task.contrastive:
            reader = ContrastiveDatasetReader()
        else:
            reader = LinspectorDatasetReader()
        # Read intrinsic vocab
        train = reader.read(os.path.join(base_path, 'train.txt'))
        dev = reader.read(os.path.join(base_path, 'dev.txt'))
        test = reader.read(os.path.join(base_path, 'test.txt'))
        return train, dev, test

    def _get_embeddings_from_model(self, probing_task):
        raise NotImplementedError

    def _get_embedding_dim(self, embeddings_file):
        with open(embeddings_file, mode='r') as file:
            line = file.readline().rstrip()
            index = line.index(' ') + 1
            vector = np.array(line[index:].replace('  ', ' ').split(' '))
            return vector.size

    def probe(self):
        metrics = dict()
        for probing_task in self.probing_tasks:
            if probing_task.contrastive:
                reader = ContrastiveDatasetReader()
            else:
                reader = LinspectorDatasetReader()
            train, dev, test = self._get_intrinsic_data(probing_task)
            vocab = Vocabulary.from_instances(train + dev)
            embeddings_file = self._get_embeddings_from_model(probing_task)
            params = Params({'embedding_dim': self._get_embedding_dim(embeddings_file), 'pretrained_file': embeddings_file, 'trainable': False})
            word_embeddings = Embedding.from_params(vocab, params=params)
            if probing_task.contrastive:
                model = ContrastiveLinear(word_embeddings, vocab)
            else:
                model = LinspectorLinear(word_embeddings, vocab)
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
            os.unlink(embeddings_file)
        return metrics

class LinspectorArchiveModel(Linspector):

    def __init__(self, language, probing_tasks, model):
        super().__init__(language, probing_tasks)
        self.model = model
        # Set default probing layer
        self.layer = 0

    def get_layers(self):
        layers = list()
        if isinstance(self.model, SimpleTagger):
            encoder = self.model.encoder._module
            modules_as_string = [str(module) for module in encoder.modules()]
            layers = [(idx, module[:module.index('(')]) for idx, module in enumerate(modules_as_string)]
        else:
            raise NotImplementedError
        return layers

    def _get_embeddings_from_model(self, probing_task):
        # Get intrinsic data for probing task
        # Set field_key to 'tokens' for SimpleTagger
        field_key='tokens'
        reader = IntrinsicDatasetReader(field_key=field_key, contrastive=probing_task.contrastive)
        base_path = os.path.join(settings.MEDIA_ROOT, 'intrinsic_data', probing_task.to_camel_case(), self.language.code)
        vocab = reader.read(base_path)
        # Select module
        module = list(self.model.encoder.modules())[self.layer]
        # Get embeddings for vocab
        embedding = torch.zeros((1, 1, module.get_input_dim()))
        def hook(module, input, output):
            # input[0] contains a torch.nn.utils.rnn.PackedSequence which also has a batch_sizes property
            embedding.copy_(input[0].data)
        handle = module.register_forward_hook(hook)
        with NamedTemporaryFile(mode='w', suffix='.vec', delete=False) as embeddings_file:
            with torch.no_grad():
                for instance in vocab:
                    token = str(instance[field_key][0])
                    self.model.forward_on_instance(instance)
                    # Write token and embedding to file
                    embeddings_file.write('{} {}\n'.format(token, ' '.join(map(str, embedding.numpy().tolist()[0][0]))))
        handle.remove()
        return embeddings_file.name

class LinspectorStaticEmbeddings(Linspector):

    def __init__(self, language, probing_tasks, embeddings_file):
        super().__init__(language, probing_tasks)
        self.embeddings_file = embeddings_file

    def _get_embeddings_from_model(self, probing_task):
        embeddings_file = NamedTemporaryFile(suffix='.vec', delete=False)
        shutil.copy2(self.embeddings_file, embeddings_file.name)
        return embeddings_file.name
