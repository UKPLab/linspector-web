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

from .models import Language, Model, ProbingTask
from .nn.dataset_readers.linspector_dataset_reader import LinspectorDatasetReader
from .nn.linspector import LinspectorArchiveModel, LinspectorStaticEmbeddings
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

class LinspectorArchiveModelTests(TestCase):

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
        linspector = LinspectorArchiveModel(language, probing_tasks, archive.model)
        self.assertGreater(len(linspector.get_layers()), 0)

    @tag('core')
    def test_get_intrinsic_data(self):
        archive = load_archive(self.archive_path)
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            linspector = LinspectorArchiveModel(language, probing_tasks, archive.model)
            for probing_task in probing_tasks:
                train, dev, test = linspector._get_intrinsic_data(probing_task)
                self.assertGreater(len(train), 0)
                self.assertGreater(len(dev), 0)
                self.assertGreater(len(test), 0)

    @tag('slow', 'core')
    def test_embeddings_file(self):
        archive = load_archive(self.archive_path)
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            linspector = LinspectorArchiveModel(language, probing_tasks, archive.model)
            for probing_task in probing_tasks:
                embeddings_file = linspector._get_embeddings_from_model(probing_task)
                self.assertTrue(os.path.isfile(embeddings_file))
                self.assertGreater(os.path.getsize(embeddings_file), 0)
                embedding_dim = linspector._get_embedding_dim(embeddings_file)
                self.assertGreater(embedding_dim, 0)
                os.unlink(embeddings_file.name)

    @tag('slow', 'core', 'nn', 'contrastive')
    def test_probe(self):
        archive = load_archive(self.archive_path)
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            linspector = LinspectorArchiveModel(language, probing_tasks, archive.model)
            metrics = linspector.probe()
            for metric in metrics.values():
                self.assertGreater(metric['accuracy'], 0)

    @tag('core', 'nn')
    def test_classifier(self):
        archive = load_archive(self.archive_path)
        # Exclude contrastive tasks
        probing_tasks = ProbingTask.objects.filter(contrastive=False)[:1]
        language = probing_tasks[0].languages.first()
        linspector = LinspectorArchiveModel(language, probing_tasks, archive.model)
        _, metric = linspector.probe().popitem()
        self.assertGreater(metric['accuracy'], 0)

    @tag('core', 'nn', 'contrastive')
    def test_contrastive_classifier(self):
        archive = load_archive(self.archive_path)
        # Filter for contrastive tasks
        probing_tasks = ProbingTask.objects.filter(contrastive=True)[:1]
        language = probing_tasks[0].languages.first()
        linspector = LinspectorArchiveModel(language, probing_tasks, archive.model)
        _, metric = linspector.probe().popitem()
        self.assertGreater(metric['accuracy'], 0)

    @tag('slow', 'nn', 'stability')
    def test_accuracy_stability(self):
        archive = load_archive(self.archive_path)
        # Get POS for German
        probing_tasks = ProbingTask.objects.filter(name='POS')[:1]
        language = Language.objects.get(code='de')
        linspector = LinspectorArchiveModel(language, probing_tasks, archive.model)
        max = 0.0
        for i in range(0, 10):
            _, metric = linspector.probe().popitem()
            if max > 0:
                # Check that accuracy does not diverge from max accuracy between iterations
                self.assertLess(abs(max - metric['accuracy']), 0.05)
                max = max(max, metric['accuracy'])
            else:
                # Set max to inital value
                max = metric['accuracy']

class LinspectorStaticEmbeddingsTests(TestCase):

    fixtures = ['languages', 'probing_tasks']

    def setUp(self):
        self.embeddings_file = os.path.join(settings.MEDIA_ROOT, 'static.vec')

    @tag('fast')
    def test_embeddings_file(self):
        self.assertTrue(os.path.isfile(self.embeddings_file))

    @tag('core', 'nn', 'static')
    def test_classifier(self):
        probing_tasks = ProbingTask.objects.filter(name='POS')[:1]
        language = probing_tasks[0].languages.get(code='de')
        linspector = LinspectorStaticEmbeddings(language, probing_tasks, self.embeddings_file)
        _, metric = linspector.probe().popitem()
        self.assertGreater(metric['accuracy'], 0)
