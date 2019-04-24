from django.core.exceptions import ObjectDoesNotExist, SuspiciousOperation
from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView

from .forms import SelectLanguageForm, SelectProbingTaskForm
from .models import Language, ProbingTask

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
        context.update(now=1, min=0, max=4)
        return context

class SelectProbingTaskView(FormView):

    template_name = 'inspector/select_probing_task.html'
    form_class = SelectProbingTaskForm

    def get_form_kwargs(self):
        # Override method to pass language parameter to SelectProbingTaskForm init
        kwargs = super(FormView, self).get_form_kwargs()
        if 'lang' in self.request.GET:
            try:
                # Verify lang parameter
                Language.objects.get(code=self.request.GET['lang'])
                kwargs['language'] = self.request.GET['lang']
            except ObjectDoesNotExist:
                raise SuspiciousOperation('No valid language selected.')
        else:
            raise SuspiciousOperation('No language selected.')
        return kwargs

    def get_success_url(self):
        return '/language/probing-task/model/?lang={}&task={}'.format(self.request.GET['lang'], ','.join(self.request.POST.getlist('probing_task')))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(now=2, min=0, max=4)
        return context
