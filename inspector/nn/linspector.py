from abc import ABC

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.modules import Embedding
from allennlp.training.util import evaluate

from django.conf import settings

import os

from inspector.models import Language, ProbingTask
from .dataset_readers.contrastive_dataset_reader import ContrastiveDatasetReader
from .dataset_readers.intrinsic_dataset_reader import IntrinsicDatasetReader
from .dataset_readers.linspector_dataset_reader import LinspectorDatasetReader
from .models.contrastive_linear import ContrastiveLinear
from .models.linspector_linear import LinspectorLinear
from .training.linspector_trainer import LinspectorTrainer

from math import floor

import numpy as np

import shutil

from tempfile import NamedTemporaryFile, TemporaryDirectory

import torch
import torch.nn as nn
import torch.optim as optim

class Linspector(ABC):

    def __init__(self, language, probing_task):
        self.language = language
        self.probing_task = probing_task
        self._callbacks = []

    def _get_intrinsic_data(self):
        base_path = os.path.join(settings.MEDIA_ROOT, 'intrinsic_data', self.probing_task.to_camel_case(), self.language.code)
        if self.probing_task.contrastive:
            reader = ContrastiveDatasetReader()
        else:
            reader = LinspectorDatasetReader()
        # Read intrinsic vocab
        train = reader.read(os.path.join(base_path, 'train.txt'))
        dev = reader.read(os.path.join(base_path, 'dev.txt'))
        test = reader.read(os.path.join(base_path, 'test.txt'))
        return train, dev, test

    def _get_embeddings_from_model(self):
        raise NotImplementedError

    def _get_embedding_dim(self, embeddings_file):
        with open(embeddings_file, mode='r') as file:
            line = file.readline().rstrip()
            index = line.index(' ') + 1
            vector = np.array(line[index:].replace('  ', ' ').split(' '))
            return vector.size

    def probe(self):
        metrics = dict()
        train, dev, test = self._get_intrinsic_data()
        # Add test data to vocabulary else evaluation will be unstable
        vocab = Vocabulary.from_instances(train + dev + test)
        embeddings_file = self._get_embeddings_from_model()
        params = Params({'embedding_dim': self._get_embedding_dim(embeddings_file), 'pretrained_file': embeddings_file, 'trainable': False})
        word_embeddings = Embedding.from_params(vocab, params=params)
        if self.probing_task.contrastive:
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
        # Use a serialization_dir otherwise evaluation uses last weights instead of best
        with TemporaryDirectory() as serialization_dir:
            trainer = LinspectorTrainer(model=model, optimizer=optimizer, iterator=iterator, train_dataset=train, validation_dataset=dev, patience=5, validation_metric='+accuracy', num_epochs=20, serialization_dir=serialization_dir, cuda_device=cuda_device, grad_clipping=5.0)
            def trainer_callback(progress):
                for callback in self._callbacks:
                    # Fill second half of progress with trainer callback
                    callback(0.5 + progress / 2)
            trainer.subscribe(trainer_callback)
            trainer.train()
            metrics = evaluate(trainer.model, test, iterator, cuda_device, batch_weight_key='')
        os.unlink(embeddings_file)
        return metrics

    def subscribe(self, callback):
        """
        Subscribe with callback to get progress [0, 1] during probing.

        Early stopping will return a value < 1.
        """
        self._callbacks.append(callback)

class LinspectorArchiveModel(Linspector):

    def __init__(self, language, probing_task, model):
        super().__init__(language, probing_task)
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

    def _get_embeddings_from_model(self):
        # Get intrinsic data for probing task
        # Set field_key to 'tokens' for SimpleTagger
        field_key='tokens'
        reader = IntrinsicDatasetReader(field_key=field_key, contrastive=self.probing_task.contrastive)
        base_path = os.path.join(settings.MEDIA_ROOT, 'intrinsic_data', self.probing_task.to_camel_case(), self.language.code)
        vocab = reader.read(base_path)
        # Select module
        module = list(self.model.encoder.modules())[self.layer]
        # Get embeddings for vocab
        embedding = torch.zeros((1, 1, module.get_input_dim()))
        def hook(module, input, output):
            # input[0] contains a torch.nn.utils.rnn.PackedSequence which also has a batch_sizes property
            embedding.copy_(input[0].data)
        handle = module.register_forward_hook(hook)
        vocab_size = len(vocab)
        callback_frequency = floor(vocab_size / 30)
        with NamedTemporaryFile(mode='w', suffix='.vec', delete=False) as embeddings_file:
            with torch.no_grad():
                for idx, instance in enumerate(vocab):
                    token = str(instance[field_key][0])
                    self.model.forward_on_instance(instance)
                    # Write token and embedding to file
                    embeddings_file.write('{} {}\n'.format(token, ' '.join(map(str, embedding.numpy().tolist()[0][0]))))
                    # Limit to max 30 callbacks to increase performance
                    # Each callback requires expensive database operations
                    # Progress accuracy is negligible
                    if idx % callback_frequency == 0:
                        for callback in self._callbacks:
                            # Fill first half of progress with embedding callback
                            callback(0.5 / vocab_size * idx)
        # Do a final callback
        for callback in self._callbacks:
            callback(0.5)
        handle.remove()
        return embeddings_file.name

class LinspectorStaticEmbeddings(Linspector):

    def __init__(self, language, probing_task, embeddings_file):
        super().__init__(language, probing_task)
        self.embeddings_file = embeddings_file

    def _get_embeddings_from_model(self):
        embeddings_file = NamedTemporaryFile(suffix='.vec', delete=False)
        shutil.copy2(self.embeddings_file, embeddings_file.name)
        return embeddings_file.name
