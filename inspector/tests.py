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

from .models import Language, Model, ProbingTask
from .nn.dataset_readers.linspector_dataset_reader import LinspectorDatasetReader
from .nn.linspector import LinspectorArchiveModel, LinspectorStaticEmbeddings
from .nn.models.linspector_linear import LinspectorLinear

import os

import random

import subprocess

from tempfile import NamedTemporaryFile, TemporaryDirectory

import torch
import torch.optim as optim

from urllib.request import urlopen

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
    """Tests AllenNLP archives.

    Test archives for different model types for a diverse set of languages and dimensions.

    Attributes:
        archives: A list of tuples containing the language code, vector dimension, and a path to an archive. For example:

            [('ar', 50, '.../media/simple_tagger.ar.d50.tar.gz')]
    """

    fixtures = ['languages', 'probing_tasks']

    def setUp(self):
        self.archives = list()
        models = ['simple_tagger', 'biaffine_parser'] # TODO: 'crf_tagger', 'esim'
        # First language in the list should have at least one contrastive and one non contrastive probing task
        langs = random.shuffle(['ar', 'hy', 'cs', 'fr', 'hu'])
        dims = random.shuffle([50, 100, 200, 300])
        for idx, model in enumerate(models):
            # Select 2 random language + dimension combinations per model
            for i in range(idx, idx * 2):
                # Use modulo to start over when the last index is reached
                lang = langs[i % (len(langs) - 1)]
                dim = dims[i % (len(dims) - 1)]
                self.archives.append((lang, dim, os.path.join(settings.MEDIA_ROOT, '{}.{}.d{}.tar.gz'.format(model, lang, dim))))

    @tag('fast')
    def test_archives(self):
        for (_, _, archive) in self.archives:
            self.assertTrue(os.path.isfile(archive[1]), msg=archive[1])
            load_archive(archive[1])

    @tag('fast', 'core')
    def test_get_layers(self):
        archive = load_archive(self.archives[0][2])
        language = Language.objects.all().first()
        probing_task = ProbingTask.objects.filter(languages__code=language.code).first()
        linspector = LinspectorArchiveModel(language, probing_task, archive.model)
        self.assertGreater(len(linspector.get_layers()), 0)

    @tag('core')
    def test_embeddings_file(self):
        for (lang, dim, archive) in self.archives:
            language = Language.objects.get(code=lang)
            probing_task = ProbingTask.objects.filter(languages__code=language.code).first()
            linspector = LinspectorArchiveModel(language, probing_task, archive.model)
            embeddings_file = linspector._get_embeddings_from_model()
            self.assertTrue(os.path.isfile(embeddings_file))
            self.assertGreater(os.path.getsize(embeddings_file), 0)
            embeddings_dim = linspector._get_embedding_dim(embeddings_file)
            self.assertEqual(embeddings_dim, dim)
            os.unlink(embeddings_file)

    @tag('core', 'nn')
    def test_classifier(self):
        archive = load_archive(self.archives[0][2])
        language = Language.objects.get(code=self.archives[0][0])
        # Exclude contrastive tasks
        probing_task = ProbingTask.objects.filter(languages__code=language.code, contrastive=False).first()
        linspector = LinspectorArchiveModel(language, probing_task, archive.model)
        metrics = linspector.probe()
        self.assertGreater(metrics['accuracy'], 0)

    @tag('core', 'nn', 'contrastive')
    def test_contrastive_classifier(self):
        archive = load_archive(self.archives[0][2])
        language = Language.objects.get(code=self.archives[0][0])
        # Exclude contrastive tasks
        probing_task = ProbingTask.objects.filter(languages__code=language.code, contrastive=True).first()
        linspector = LinspectorArchiveModel(language, probing_task, archive.model)
        metrics = linspector.probe()
        self.assertGreater(metrics['accuracy'], 0)

    @tag('slow', 'nn', 'consistency')
    def test_accuracy_consistency(self):
        archive = load_archive(self.archives[0][2])
        language = Language.objects.get(code=self.archives[0][0])
        probing_task = ProbingTask.objects.filter(languages__code=language.code).first()
        accuracy = 0.0
        for i in range(0, 3):
            metrics = linspector.probe()
            if accuracy > 0:
                # Check that accuracy does not diverge between iterations (with tolerance)
                self.assertLess(abs(accuracy - metrics['accuracy']), 0.01)
                accuracy = max(accuracy, metrics['accuracy'])
            else:
                # Set accuracy to inital value
                accuracy = metrics['accuracy']

    @tag('slow', 'core', 'nn', 'contrastive')
    def test_probe_all(self):
        for (lang, dim, archive) in self.archives:
            language = Language.objects.get(code=lang)
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            for probing_task in probing_tasks:
                linspector = LinspectorArchiveModel(language, probing_task, archive.model)
                metrics = linspector.probe()
                self.assertGreater(metrics['accuracy'], 0)

class LinspectorStaticEmbeddingsTests(TestCase):
    """Tests static embeddings.

    Loads pretrained fastText embeddings for a diverse set of languages and dimensions.

    Attributes:
        files: A list of tuples containing the language code and a path to an embeddings file. For example:

            [('ar', '.../media/fasttext.ar.vec')]
    """

    fixtures = ['languages', 'probing_tasks']

    def setUp(self):
        self.files = list()
        langs = ['ar', 'hy', 'cs', 'fr', 'hu']
        for lang in langs:
            path = os.path.join(settings.MEDIA_ROOT, 'fasttext.{}.vec'.format(lang))
            # Check if file already exists
            if not os.path.isfile(path):
                response = urlopen('https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'.format(lang))
                with open(path, 'wb+') as file:
                    while True:
                        chunk = response.read(16 * 1024)
                        if chunk:
                            file.write(chunk)
                        else:
                            break
            self.files.append((lang, path))

    @tag('core', 'static')
    def test_get_embedding_dim(self):
        for (lang, fasttext) in self.files:
            language = Language.objects.get(code=lang)
            probing_task = ProbingTask.objects.filter(languages__code=language.code).first()
            linspector = LinspectorStaticEmbeddings(language, probing_task, fasttext)
            embeddings_dim = linspector._get_embedding_dim(fasttext)
            # All pretrained fastText files should have dim 300
            self.assertEqual(embeddings_dim, 300)

    @tag('core', 'static')
    def test_embeddings_file(self):
        for (lang, fasttext) in self.files:
            language = Language.objects.get(code=lang)
            probing_task = ProbingTask.objects.filter(languages__code=language.code).first()
            linspector = LinspectorStaticEmbeddings(language, probing_task, fasttext)
            embeddings_file = linspector._get_embeddings_from_model()
            self.assertTrue(os.path.isfile(embeddings_file))
            self.assertGreater(os.path.getsize(embeddings_file), 0)
            embeddings_dim = linspector._get_embedding_dim(embeddings_file)
            self.assertEqual(embeddings_dim, 300)
            os.unlink(embeddings_file)

    @tag('slow', 'core', 'nn', 'static')
    def test_probe_all(self):
        for (lang, fasttext) in self.files:
            language = Language.objects.get(code=lang)
            probing_tasks = ProbingTask.objects.filter(languages__code=language.code)
            for probing_task in probing_tasks:
                linspector = LinspectorStaticEmbeddings(language, probing_task, fasttext)
                metrics = linspector.probe()
                self.assertGreater(metrics['accuracy'], 0)
