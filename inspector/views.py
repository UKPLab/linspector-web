from allennlp.models.archival import load_archive

import ast

from celery.result import AsyncResult

from collections import defaultdict

from copy import deepcopy

from django.conf import settings
from django.core.exceptions import SuspiciousOperation
from django.http import JsonResponse
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView

from django_celery_results.models import TaskResult

import json

import os

from .forms import SelectLanguageForm, SelectLayerForm, SelectProbingTaskForm, UploadEpochForm, UploadModelForm
from .models import Language, Model, ProbingTask
from .nn.linspector import LinspectorArchiveModel
from .nn.utils import Classifier
from .tasks import probe
from .utils import get_request_params

class IndexView(TemplateView):

    template_name = 'inspector/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['classifiers'] = list(map(str, Classifier))
        context['languages'] = Language.objects.order_by('name')
        context['probing_tasks'] = ProbingTask.objects.order_by('name')
        return context

class AboutView(TemplateView):

    template_name = 'inspector/about.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['case_marking'] = ProbingTask.objects.get(name='Case Marking').languages.all().order_by('name')
        context['gender'] = ProbingTask.objects.get(name='Gender').languages.all().order_by('name')
        context['person'] = ProbingTask.objects.get(name='Person').languages.all().order_by('name')
        context['tense'] = ProbingTask.objects.get(name='Tense').languages.all().order_by('name')
        context['shared'] = ProbingTask.objects.get(name='Shared Morphological Feature').languages.all().order_by('name')
        context['odd'] = ProbingTask.objects.get(name='Odd Morphological Feature').languages.all().order_by('name')
        context['pseudoword'] = ProbingTask.objects.get(name='Pseudoword').languages.all().order_by('name')
        return context

class SelectLanguageView(FormView):

    template_name = 'inspector/select_language.html'
    form_class = SelectLanguageForm

    def get_success_url(self):
        return 'probing-task/?lang={}'.format(self.request.POST['language'])

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['back'] = '../'
        return context

class SelectProbingTaskView(FormView):

    template_name = 'inspector/select_probing_task.html'
    form_class = SelectProbingTaskForm

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        if 'lang' not in request.GET:
            raise SuspiciousOperation('`lang` parameter is missing.')
        else:
            self._language = get_request_params(request)

    def get_form_kwargs(self):
        # Override method to pass language parameter to SelectProbingTaskForm init
        kwargs = super().get_form_kwargs()
        kwargs['language'] = self._language.code
        return kwargs

    def get_success_url(self):
        return 'model/?lang={}&task={}'.format(self._language.code, ','.join(self.request.POST.getlist('probing_task')))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['back'] = '../'
        return context

class UploadResponseMixin:

    def form_invalid(self, form):
        response = super().form_invalid(form)
        if self.request.is_ajax():
            return JsonResponse(form.errors, status=400)
        else:
            return response

    def form_valid(self, form):
        response = super().form_valid(form)
        if self.request.is_ajax():
            data = {
                'url': self.get_success_url()
            }
            return JsonResponse(data)
        else:
            return response

class UploadModelView(UploadResponseMixin, FormView):

    template_name = 'inspector/upload_model.html'
    form_class = UploadModelForm

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        if 'lang' not in request.GET:
            raise SuspiciousOperation('`lang` parameter is missing.')
        elif 'task' not in request.GET:
            raise SuspiciousOperation('`task` parameter is missing.')
        else:
            self._language, self._probing_tasks = get_request_params(request)

    def get_success_url(self):
        _, extension = os.path.splitext(self._model.upload.name)
        if extension == '.gz':
            url = 'epoch/?lang={}&task={}&model={}'
        else:
            # Skip layer selection for static embeddings
            url = 'probe/?lang={}&task={}&model={}'
        return url.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]), self._model.id)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['back'] = '../?lang={}'.format(self._language.code)
        return context

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        if form.is_valid():
            self._model = Model(name=request.FILES['model'].name[:35], upload=request.FILES['model'])
            self._model.save()
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

class UploadEpochView(UploadResponseMixin, FormView):

    template_name = 'inspector/upload_epoch.html'
    form_class = UploadEpochForm

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        if 'lang' not in request.GET:
            raise SuspiciousOperation('`lang` parameter is missing.')
        elif 'task' not in request.GET:
            raise SuspiciousOperation('`task` parameter is missing.')
        elif 'model' not in request.GET:
            raise SuspiciousOperation('`model` parameter is missing.')
        else:
            self._language, self._probing_tasks, self._model = get_request_params(request)
            # Cleanup of previous uploaded epochs
            # Otherwise the user could upload additional epochs using the back button
            for epoch in self._model.epoch.all():
                # Remove before delete so the deletion won't cascade to the model
                self._model.epoch.remove(epoch)
                epoch.delete()

    def get_success_url(self):
        return 'layer/?lang={}&task={}&model={}'.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]), self._model.id)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['back'] = '../?lang={}&task={}'.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]))
        context['skip'] = self.get_success_url()
        return context

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('epoch')
        if form.is_valid():
            for file in files:
                self._model.epoch.create(name=file.name[:35], upload=file)
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

class SelectLayerView(FormView):

    template_name = 'inspector/select_layer.html'
    form_class = SelectLayerForm

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        if 'lang' not in request.GET:
            raise SuspiciousOperation('`lang` parameter is missing.')
        elif 'task' not in request.GET:
            raise SuspiciousOperation('`task` parameter is missing.')
        elif 'model' not in request.GET:
            raise SuspiciousOperation('`model` parameter is missing.')
        else:
            self._language, self._probing_tasks, self._model = get_request_params(request)

    def get_form_kwargs(self):
        # Override method to pass language parameter to SelectLayerForm init
        kwargs = super().get_form_kwargs()
        archive = load_archive(self._model.upload.path)
        linspector = LinspectorArchiveModel(self._language, self._probing_tasks.first(), archive.model)
        kwargs['layer'] = linspector.get_layers()
        return kwargs

    def get_success_url(self):
        return 'probe/?lang={}&task={}&model={}&layer={}'.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]), self._model.id, self.request.POST['layer'])

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['back'] = '../?lang={}&task={}&model={}'.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]), self._model.id)
        return context

class ProbeView(TemplateView):

    template_name = 'inspector/probe.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        if request.is_ajax():
            if 'id' not in request.GET:
                raise SuspiciousOperation('`id` parameter is missing.')
            else:
                self._result = AsyncResult(request.GET['id'])
        elif 'lang' not in request.GET:
            raise SuspiciousOperation('`lang` parameter is missing.')
        elif 'task' not in request.GET:
            raise SuspiciousOperation('`task` parameter is missing.')
        elif 'model' not in request.GET:
            raise SuspiciousOperation('`model` parameter is missing.')
        elif 'layer' not in request.GET:
            # Skip layer selection for static embeddings
            language, probing_tasks, model = get_request_params(request)
            _, extension = os.path.splitext(model.upload.name)
            if extension == '.gz':
                raise SuspiciousOperation('`layer` parameter is missing.')
            self._task = probe.delay(language.pk, [probing_task.pk for probing_task in probing_tasks], model.id)
        else:
            language, probing_tasks, model, layer = get_request_params(request)
            self._task = probe.delay(language.pk, [probing_task.pk for probing_task in probing_tasks], model.id, layer)

    def dispatch(self, request, *args, **kwargs):
        if request.is_ajax():
            data = {
                'state': self._result.state,
                'info': self._result.info
            }
            if self._result.state == 'SUCCESS':
                data['url'] = 'result/?id={}'.format(self._result.id)
            return JsonResponse(data)
        else:
            return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['task_id'] = self._task.task_id
        return context

class ShowResultView(TemplateView):

    template_name = 'inspector/show_result.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        if 'id' in request.GET:
            self._result = TaskResult.objects.get(task_id=request.GET['id'])
        elif 'id' in kwargs:
            self._result = TaskResult.objects.get(task_id=str(kwargs['id']))
        else:
            raise SuspiciousOperation('`id` parameter is missing.')
        if self._result.status != 'SUCCESS':
            raise RuntimeError('Task is not ready.')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        metrics = json.loads(self._result.result)
        context['probing_tasks'] = metrics
        file_names = defaultdict(dict)
        for probing_task, file in metrics.items():
            for file_name, metrics in file.items():
                file_names[file_name][probing_task] = metrics
        context['file_names'] = dict(file_names)
        # Create a second dict with the difference between each epoch and best metrics
        if len(file_names) > 1:
            diff = deepcopy(file_names)
            del diff['best']
            for file_name, probing_tasks in diff.items():
                for probing_task, metrics in probing_tasks.items():
                    for metric, value in metrics.items():
                        diff[file_name][probing_task][metric] = value - file_names['best'][probing_task][metric]
            context['diff'] = dict(diff)
        args = ast.literal_eval(self._result.task_args)
        language = Language.objects.get(pk=args[0])
        context['language'] = language.name
        context['date'] = self._result.date_done
        context['share'] = self.request.build_absolute_uri('/{}/'.format(self._result.task_id))
        context['debug'] = settings.DEBUG
        return context
