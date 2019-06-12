from allennlp.models.archival import load_archive

from celery import current_task, shared_task

from collections import defaultdict

from .models import Language, Model, ProbingTask
from .nn.linspector import LinspectorArchiveModel, LinspectorStaticEmbeddings

import os

@shared_task
def probe(language, probing_tasks, model, layer = None):
    # Requires Eventlet as a celery execution pool
    language = Language.objects.get(pk=language)
    probing_tasks = ProbingTask.objects.filter(pk__in=probing_tasks).order_by('name')
    model = Model.objects.get(id=model)
    # All epochs + best weights
    total_epochs = model.epoch.all().count() + 1
    total = len(probing_tasks) * total_epochs
    metrics = defaultdict(dict)
    for idx, probing_task in enumerate(probing_tasks):
        current_task.update_state(state='PROGRESS', meta={'progress': 1 / total * idx * total_epochs, 'task': probing_task.name})
        if layer is not None:
            # Probe archive model
            archive = load_archive(model.upload.path)
            linspector = LinspectorArchiveModel(language, probing_task, archive.model)
            linspector.layer = layer
        else:
            # Probe static embeddings
            linspector = LinspectorStaticEmbeddings(language, probing_task, model.upload.path)
        current_epoch = 0
        def callback(progress):
            current_task.update_state(state='PROGRESS', meta={'progress': 1 / total * (idx * total_epochs + current_epoch + progress), 'task': probing_task.name})
        linspector.subscribe(callback)
        metrics[probing_task.name]['best'] = linspector.probe()
        for epoch in model.epoch.all():
            current_epoch += 1
            archive = load_archive(model.upload.path, weights_file=epoch.upload.path)
            linspector.model = archive.model
            metrics[probing_task.name][os.path.splitext(epoch.name)[0]] = linspector.probe()
    # Delete model
    model.delete()
    # Sort keys by accuracy descending
    map = sorted(metrics, key=lambda i: metrics[i][model.name]['accuracy'], reverse=True)
    # Use key map to create a sorted dict
    return {key: metrics[key] for key in map}
