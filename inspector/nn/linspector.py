from abc import ABC

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.training.util import evaluate

from django.conf import settings

import inspect

import os

from .dataset_readers.contrastive_dataset_reader import ContrastiveDatasetReader
from .dataset_readers.intrinsic_dataset_reader import IntrinsicDatasetReader
from .dataset_readers.linspector_dataset_reader import LinspectorDatasetReader
from .models.contrastive_linear import ContrastiveLinear
from .models.linspector_linear import LinspectorLinear
from .training.linspector_trainer import LinspectorTrainer

from math import floor

from tempfile import NamedTemporaryFile, TemporaryDirectory

import torch
import torch.optim as optim

class Linspector(ABC):
    """Abstract base class for probing embeddings.

    Attributes:
        language: A Language model specifying the embedding language.
        probing_tasks: A ProbingTask model specifying the probing task and classifier type.
    """

    def __init__(self, language, probing_task):
        self.language = language
        self.probing_task = probing_task
        self._callbacks = []

    def _get_intrinsic_data(self):
        """Returns intrinsic data for the current probing task and language.

        Returns:
            A tuple containing iterables for train, dev, and test data.
        """
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
        """Retrieves embeddings and writes them to a NamedTemporaryFile.

        Returns:
            A path as a str to the NamedTemporaryFile embeddings file. The embeddings file should contain a lowercased token followed by a vector separated by whitespace. For example:

            katapultiert 0.019 -0.29 -0.34 0.076 ...
            rutsch 0.019 -0.29 -0.34 0.076 ...
            ...
        """
        raise NotImplementedError

    def _get_embedding_dim(self, embeddings_file):
        """Get the embedding dimension from an embeddings file.

        Args:
            embeddings_file: Path as str to an embeddings file.

        Returns:
            Embedding dimension as an int.
        """
        dim = 0
        with open(embeddings_file, mode='r') as file:
            for line in file:
                split = line.strip().split()
                # Find beginning of vector
                for idx, value in enumerate(split):
                    # Skip token
                    if idx > 0:
                        try:
                            float(value)
                            dim = max(dim, len(split[idx:]))
                            break
                        except ValueError:
                            # Disgard potential artefacts
                            continue
        return dim

    def probe(self):
        """Probes an embeddings file and returns its metrics.

        Trains a linear model using embeddings from _get_embeddings_from_model() as a pretrained embeddings layer and intrinsic data for the current probing task.

        Returns:
            A dict containing metrics from allennlp.training.util.evaluate.
        """
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
        """Subscribe with callback to get progress [0, 1] during probing.

        Early stopping will return a value < 1.

        Args:
            callback: A function taking an float between 0 and 1 as input.
        """
        self._callbacks.append(callback)

class LinspectorArchiveModel(Linspector):
    """Probes AllenNLP models.

    Attributes:
        language: A Language model specifying the embedding language.
        probing_tasks: A ProbingTask model specifying the probing task and classifier type.
        model: An allennlp.models.model to probe. The model has to be a vanilla AllenNLP model. Custom models are not supported.
        layer: Key of probing layer.
    """

    def __init__(self, language, probing_task, model):
        super().__init__(language, probing_task)
        self.model = model
        self.layer = None

    def get_layers(self):
        """Returns a list of layers available for probing.

        Returns:
            A list of tuples containing a dict key and the layer display name.
        """
        layers = list()
        for name, module in self.model.named_children():
            if isinstance(module, PytorchSeq2SeqWrapper) or isinstance(module, PytorchSeq2VecWrapper):
                # Get name from wrapped class e.g. LSTM
                layers.append((name, module._module.__class__.__name__))
            elif isinstance(module, Seq2SeqEncoder) or isinstance(module, Seq2VecEncoder):
                layers.append((name, name))
        return layers

    def _get_embeddings_from_model(self):
        # Get intrinsic data for probing task
        # Set field_key to first argument name of forward method
        field_key = inspect.getfullargspec(self.model.forward)[0][1]
        reader = IntrinsicDatasetReader(field_key=field_key, contrastive=self.probing_task.contrastive)
        base_path = os.path.join(settings.MEDIA_ROOT, 'intrinsic_data', self.probing_task.to_camel_case(), self.language.code)
        vocab = reader.read(base_path)
        # Select module
        if self.layer is not None:
            module = self.model._modules[self.layer]
        else:
            # If no layer is specified select the first one
            layers = self.get_layers()
            module = self.model._modules[layers[0][0]]
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
    """Probes static embedding files.

    Attributes:
        language: A Language model specifying the embedding language.
        probing_tasks: A ProbingTask model specifying the probing task and classifier type.
        embeddings_file: Path as str to an embeddings file. The file should contain a token followed by a vector separated by whitespace. For example:

            katapultiert 0.019 -0.29 -0.34 0.076 ...
            rutsch 0.019 -0.29 -0.34 0.076 ...
            ...
    """

    def __init__(self, language, probing_task, embeddings_file):
        super().__init__(language, probing_task)
        self.embeddings_file = embeddings_file

    def _get_embeddings_from_model(self):
        dim = self._get_embedding_dim(self.embeddings_file)
        with NamedTemporaryFile(mode='w', suffix='.vec', delete=False) as embeddings_file:
            with open(self.embeddings_file) as data:
                file_size = sum(1 for line in data)
                callback_frequency = floor(file_size / 30)
                data.seek(0)
                for idx, line in enumerate(data):
                    split = line.strip().split()
                    # Disgard lines of size other than embeddings dim plus token
                    if len(split) == dim + 1:
                        # Lowercase tokens
                        token = split[0].lower()
                        embedding = split[-dim:]
                        # Discard non alphabetic tokens
                        if token.isalpha():
                            # Write token and embedding to file
                            embeddings_file.write('{} {}\n'.format(token, ' '.join(embedding)))
                    # Limit to max 30 callbacks to increase performance
                    # Each callback requires expensive database operations
                    # Progress accuracy is negligible
                    if idx % callback_frequency == 0:
                        for callback in self._callbacks:
                            # Fill first half of progress with embedding callback
                            callback(0.5 / file_size * idx)
        # Do a final callback
        for callback in self._callbacks:
            callback(0.5)
        return embeddings_file.name
