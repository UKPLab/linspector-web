from django.forms import CheckboxSelectMultiple, Form, ModelChoiceField, ModelMultipleChoiceField, Select

from .models import Language, ProbingTask

class SelectLanguageForm(Form):

    language = ModelChoiceField(queryset=Language.objects.order_by('name'), required=True, label='Language', label_suffix='', to_field_name='code', widget=Select(attrs={'class': 'custom-select'}))

class SelectProbingTaskForm(Form):

    def __init__(self, language, *args, **kwargs):
        # Override method to filter ProbingTask by language parameter
        super().__init__(*args, **kwargs)
        self.fields['probing_task'] = ModelMultipleChoiceField(queryset=ProbingTask.objects.filter(languages__code=language).order_by('name'), required=True, label='Probing Tasks', label_suffix='', widget=CheckboxSelectMultiple(attrs={'class': 'form-check-input'}))
