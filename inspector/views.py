from allennlp.models.archival import load_archive

from django.core.exceptions import SuspiciousOperation
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView

import os

from .forms import SelectLanguageForm, SelectLayerForm, SelectProbingTaskForm, UploadModelForm
from .models import Language, Model, ProbingTask
from .nn.linspector import LinspectorArchiveModel, LinspectorStaticEmbeddings
from .utils import get_request_params

class IndexView(TemplateView):

    template_name = 'inspector/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['languages'] = Language.objects.order_by('name')
        context['probing_tasks'] = ProbingTask.objects.order_by('name')
        return context

class SelectLanguageView(FormView):

    template_name = 'inspector/select_language.html'
    form_class = SelectLanguageForm

    def get_success_url(self):
        return '/language/probing-task/?lang={}'.format(self.request.POST['language'])

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(back='../', now=0, min=0, max=4)
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
        return '/language/probing-task/model/?lang={}&task={}'.format(self._language.code, ','.join(self.request.POST.getlist('probing_task')))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(back='../', now=1, min=0, max=4)
        return context

class UploadModelResponseMixin:

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

class UploadModelView(UploadModelResponseMixin, FormView):

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
            url = '/language/probing-task/model/layer/?lang={}&task={}&model={}'
        else:
            # Skip layer selection for static embeddings
            url = '/language/probing-task/model/layer/probe/?lang={}&task={}&model={}'
        return url.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]), self._model.id)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['back'] = '../?lang={}'.format(self._language.code)
        context.update(now=2, min=0, max=4)
        return context

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        if form.is_valid():
            self._model = Model(upload=request.FILES['model'])
            self._model.save()
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
        linspector = LinspectorArchiveModel(self._language, self._probing_tasks, archive.model)
        kwargs['layer'] = linspector.get_layers()
        return kwargs

    def get_success_url(self):
        return '/language/probing-task/model/layer/probe/?lang={}&task={}&model={}&layer={}'.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]), self._model.id, self.request.POST['layer'])

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['back'] = '../?lang={}&task={}'.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]))
        context.update(now=3, min=0, max=4)
        return context

class ProbeView(TemplateView):

    template_name = 'inspector/probe.html'

    def setup(self, request, *args, **kwargs):
        super().setup(request, *args, **kwargs)
        if 'lang' not in request.GET:
            raise SuspiciousOperation('`lang` parameter is missing.')
        elif 'task' not in request.GET:
            raise SuspiciousOperation('`task` parameter is missing.')
        elif 'model' not in request.GET:
            raise SuspiciousOperation('`model` parameter is missing.')
        elif 'layer' not in request.GET:
            # Skip layer selection for static embeddings
            self._language, self._probing_tasks, self._model = get_request_params(request)
            _, extension = os.path.splitext(self._model.upload.name)
            if extension == '.gz':
                raise SuspiciousOperation('`layer` parameter is missing.')
        else:
            self._language, self._probing_tasks, self._model, self._layer = get_request_params(request)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if hasattr(self, '_layer'):
            # Probe archive model
            archive = load_archive(self._model.upload.path)
            linspector = LinspectorArchiveModel(self._language, self._probing_tasks, archive.model)
            linspector.layer = self._layer
            metrics = linspector.probe()
        else:
            # Probe static embeddings
            linspector = LinspectorStaticEmbeddings(self._language, self._probing_tasks, self._model.upload.path)
            metrics = linspector.probe()
        # Sort keys by accuracy descending
        map = sorted(metrics, key=lambda i: metrics[i]['accuracy'], reverse=True)
        # Use key map to create a sorted dict
        context['metrics'] = {key: metrics[key] for key in map}
        self._model.upload.delete()
        self._model.delete()
        return context
