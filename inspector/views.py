from allennlp.models.archival import load_archive

from django.core.exceptions import SuspiciousOperation
from django.core.files.storage import FileSystemStorage
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView

from .forms import SelectLanguageForm, SelectLayerForm, SelectProbingTaskForm, UploadModelForm
from .linspector import LinspectorModel
from .models import Language, Model, ProbingTask
from .utils import get_request_params

# Create your views here.
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
        context.update(now=0, min=0, max=4)
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
        context.update(now=1, min=0, max=4)
        return context

class UploadModelView(FormView):

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
        return '/language/probing-task/model/layer/?lang={}&task={}&model={}'.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]), self._model.id)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
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
        linspector = LinspectorModel(self._language, self._probing_tasks, archive.model)
        kwargs['layer'] = linspector.get_layers()
        return kwargs

    def get_success_url(self):
        return '/language/probing-task/model/layer/probe/?lang={}&task={}&model={}&layer={}'.format(self._language.code, ','.join([str(task.id) for task in self._probing_tasks]), self._model.id, self.request.POST['layer'])

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
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
            raise SuspiciousOperation('`layer` parameter is missing.')
        else:
            self._language, self._probing_tasks, self._model, self._layer = get_request_params(request)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        archive = load_archive(self._model.upload.path)
        linspector = LinspectorModel(self._language, self._probing_tasks, archive.model)
        metrics = linspector.probe(self._layer)
        # Sort keys by accuracy descending
        map = sorted(metrics, key=lambda i: metrics[i]['accuracy'], reverse=True)
        # Use key map to create a sorted dict
        context['metrics'] = {key: metrics[key] for key in map}
        self._model.upload.delete()
        self._model.delete()
        return context
