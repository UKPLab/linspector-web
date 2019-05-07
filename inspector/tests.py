from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.models.archival import load_archive
from allennlp.modules import Embedding
from allennlp.training.trainer import Trainer

from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from django.test import tag, TestCase

import json

import os

from .linspector import LinspectorModel
from .models import Language, Model, ProbingTask
from .nn.dataset_readers.linspector_dataset_reader import LinspectorDatasetReader
from .nn.models.linspector_linear import LinspectorLinear

import subprocess

from tempfile import NamedTemporaryFile, TemporaryDirectory

import torch
import torch.optim as optim

class LinspectorTest(TestCase):

    fixtures = ['languages', 'probing_tasks']

    @tag('fast', 'core')
    def test_language_exists(self):
        languages = Language.objects.all()
        self.assertTrue(languages.exists())

    @tag('fast', 'core')
    def test_probing_task_for_each_language(self):
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            self.assertTrue(probing_tasks.exists())

class LinspectorModelTests(TestCase):

    fixtures = ['languages', 'probing_tasks']

    def setUp(self):
        self.archive_path = os.path.join(settings.MEDIA_ROOT, 'model.tar.gz')

    @tag('fast')
    def test_archive(self):
        self.assertTrue(os.path.isfile(self.archive_path))
        archive = load_archive(self.archive_path)

    @tag('fast')
    def test_get_layers(self):
        archive = load_archive(self.archive_path)
        language = Language.objects.all().first()
        probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
        linspector = LinspectorModel(language, probing_tasks, archive.model)
        self.assertGreater(len(linspector.get_layers()), 0)

    @tag('core')
    def test_get_intrinsic_data(self):
        archive = load_archive(self.archive_path)
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            linspector = LinspectorModel(language, probing_tasks, archive.model)
            for probing_task in probing_tasks:
                reader = LinspectorDatasetReader(field_key='tokens')
                train, dev, test = linspector._get_intrinsic_data(probing_task, reader)
                self.assertGreater(len(train), 0)
                self.assertGreater(len(dev), 0)
                self.assertGreater(len(test), 0)

    @tag('slow')
    def test_embeddings_file(self):
        archive = load_archive(self.archive_path)
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            linspector = LinspectorModel(language, probing_tasks, archive.model)
            for probing_task in probing_tasks:
                embeddings_file = linspector._get_embeddings_from_model(probing_task, 0)
                self.assertTrue(os.path.isfile(embeddings_file.name))
                self.assertGreater(os.path.getsize(embeddings_file.name), 0)
                embedding_dim = linspector._get_embedding_dim(embeddings_file)
                self.assertGreater(embedding_dim, 0)
                os.unlink(embeddings_file.name)

    @tag('slow', 'core', 'nn', 'contrastive')
    def test_probe(self):
        archive = load_archive(self.archive_path)
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            linspector = LinspectorModel(language, probing_tasks, archive.model)
            metrics = linspector.probe(layer=0)
            for metric in metrics.values():
                self.assertGreater(metric['accuracy'], 0)

    @tag('core', 'nn')
    def test_classifier(self):
        archive = load_archive(self.archive_path)
        # Exclude contrastive tasks
        probing_tasks = ProbingTask.objects.exclude(name__contains='Feat')[:1]
        language = probing_tasks[0].languages.first()
        linspector = LinspectorModel(language, probing_tasks, archive.model)
        _, metric = linspector.probe(layer=0).popitem()
        self.assertGreater(metric['accuracy'], 0)

    @tag('core', 'nn', 'contrastive')
    def test_contrastive_classifier(self):
        archive = load_archive(self.archive_path)
        # Filter for contrastive tasks
        probing_tasks = ProbingTask.objects.filter(name__contains='Feat')[:1]
        language = probing_tasks[0].languages.first()
        linspector = LinspectorModel(language, probing_tasks, archive.model)
        _, metric = linspector.probe(layer=0).popitem()
        self.assertGreater(metric['accuracy'], 0)
