from allennlp.models.archival import load_archive

from celery import current_task, shared_task

from .models import Language, Model, ProbingTask
from .nn.linspector import LinspectorArchiveModel, LinspectorStaticEmbeddings

@shared_task
def probe(language, probing_tasks, model, layer = None):
    # Requires eventlet as a celery execution pool
    language = Language.objects.get(pk=language)
    probing_tasks = ProbingTask.objects.filter(pk__in=probing_tasks).order_by('name')
    model = Model.objects.get(id=model)
    total = len(probing_tasks)
    metrics = dict()
    for idx, probing_task in enumerate(probing_tasks):
        current_task.update_state(state='PROGRESS', meta={'progress': 1 / total * idx, 'task': probing_task.name})
        if layer is not None:
            # Probe archive model
            archive = load_archive(model.upload.path)
            linspector = LinspectorArchiveModel(language, probing_task, archive.model)
            linspector.layer = layer
        else:
            # Probe static embeddings
            linspector = LinspectorStaticEmbeddings(language, probing_task, model.upload.path)
        def callback(progress):
            current_task.update_state(state='PROGRESS', meta={'progress': 1 / total * (idx + progress), 'task': probing_task.name})
        linspector.subscribe(callback)
        metrics[probing_task.name] = linspector.probe()
    # Delete model
    model.delete()
    # Sort keys by accuracy descending
    map = sorted(metrics, key=lambda i: metrics[i]['accuracy'], reverse=True)
    # Use key map to create a sorted dict
    return {key: metrics[key] for key in map}
