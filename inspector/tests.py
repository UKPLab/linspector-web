from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.models.archival import load_archive
from allennlp.modules import Embedding
from allennlp.training.trainer import Trainer

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
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

class IntrinsicDataTest(TestCase):

    fixtures = ['languages', 'probing_tasks']

    @tag('fast', 'core', 'fixtures')
    def test_intrinsic_data_for_fixture(self):
        files = ['train.txt', 'dev.txt', 'test.txt']
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            for probing_task in probing_tasks:
                for file in files:
                    path = os.path.join(settings.MEDIA_ROOT, 'intrinsic_data', probing_task.to_camel_case(), language.code, file)
                    self.assertTrue(os.path.isfile(path), msg=path)

    @tag('fast', 'core', 'fixtures')
    def test_fixture_for_intrinsic_data(self):
        path = os.path.join(settings.MEDIA_ROOT, 'intrinsic_data')
        # Map from probing task directory format (camel case) to id
        map = {probing_task.to_camel_case(): probing_task.id for probing_task in ProbingTask.objects.all()}
        with os.scandir(path) as probing_tasks:
            for probing_task in probing_tasks:
                if probing_task.is_dir():
                    self.assertTrue(probing_task.name in map, msg=probing_task.path)
                    with os.scandir(probing_task.path) as languages:
                        for language in languages:
                            if language.is_dir():
                                self.assertTrue(ProbingTask.objects.filter(languages__code=language.name, id=map[probing_task.name]).exists(), msg=language.path)

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
        linspector = LinspectorArchiveModel(language, probing_tasks.first(), archive.model)
        self.assertGreater(len(linspector.get_layers()), 0)

    @tag('core')
    def test_get_intrinsic_data(self):
        archive = load_archive(self.archive_path)
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            for probing_task in probing_tasks:
                linspector = LinspectorArchiveModel(language, probing_task, archive.model)
                train, dev, test = linspector._get_intrinsic_data()
                self.assertGreater(len(train), 0)
                self.assertGreater(len(dev), 0)
                self.assertGreater(len(test), 0)

    @tag('slow', 'core')
    def test_embeddings_file(self):
        archive = load_archive(self.archive_path)
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            for probing_task in probing_tasks:
                linspector = LinspectorArchiveModel(language, probing_task, archive.model)
                embeddings_file = linspector._get_embeddings_from_model()
                self.assertTrue(os.path.isfile(embeddings_file))
                self.assertGreater(os.path.getsize(embeddings_file), 0)
                embedding_dim = linspector._get_embedding_dim(embeddings_file)
                self.assertGreater(embedding_dim, 0)
                os.unlink(embeddings_file)

    @tag('slow', 'core', 'nn', 'contrastive')
    def test_probe(self):
        archive = load_archive(self.archive_path)
        languages = Language.objects.all()
        for language in languages:
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            for probing_task in probing_tasks:
                linspector = LinspectorArchiveModel(language, probing_task, archive.model)
                metrics = linspector.probe()
                self.assertTrue(isinstance(metrics['accuracy'], float))

    @tag('core', 'nn')
    def test_classifier(self):
        archive = load_archive(self.archive_path)
        # Exclude contrastive tasks
        probing_task = ProbingTask.objects.filter(contrastive=False).first()
        language = probing_task.languages.first()
        linspector = LinspectorArchiveModel(language, probing_task, archive.model)
        metrics = linspector.probe()
        self.assertGreater(metrics['accuracy'], 0)

    @tag('core', 'nn', 'contrastive')
    def test_contrastive_classifier(self):
        archive = load_archive(self.archive_path)
        # Filter for contrastive tasks
        probing_task = ProbingTask.objects.filter(contrastive=True).first()
        language = probing_task.languages.first()
        linspector = LinspectorArchiveModel(language, probing_task, archive.model)
        metrics = linspector.probe()
        self.assertGreater(metrics['accuracy'], 0)

    @tag('slow', 'nn', 'stability')
    def test_accuracy_stability(self):
        archive = load_archive(self.archive_path)
        # Get POS for German
        probing_task = ProbingTask.objects.filter(name='POS').first()
        language = Language.objects.get(code='de')
        linspector = LinspectorArchiveModel(language, probing_task, archive.model)
        max_accuracy = 0.0
        for i in range(0, 10):
            metrics = linspector.probe()
            if max_accuracy > 0:
                # Check that accuracy does not diverge from max accuracy between iterations (with tolerance)
                self.assertLess(abs(max_accuracy - metrics['accuracy']), 0.05)
                max_accuracy = max(max_accuracy, metrics['accuracy'])
            else:
                # Set max accuracy to inital value
                max_accuracy = metrics['accuracy']

class LinspectorStaticEmbeddingsTests(TestCase):

    fixtures = ['languages', 'probing_tasks']

    def setUp(self):
        self.embeddings_file = os.path.join(settings.MEDIA_ROOT, 'static.vec')

    @tag('fast', 'static')
    def test_embeddings_file(self):
        self.assertTrue(os.path.isfile(self.embeddings_file))

    @tag('core', 'nn', 'static')
    def test_classifier(self):
        probing_task = ProbingTask.objects.filter(name='POS').first()
        language = probing_task.languages.get(code='de')
        linspector = LinspectorStaticEmbeddings(language, probing_task, self.embeddings_file)
        metrics = linspector.probe()
        self.assertGreater(metrics['accuracy'], 0)
