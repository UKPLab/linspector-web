from django.core.exceptions import ValidationError
from django.forms import CheckboxSelectMultiple, ChoiceField, FileField, FileInput, Form, ModelChoiceField, ModelMultipleChoiceField, Select

import os

from .models import Language, ProbingTask

class SelectLanguageForm(Form):

    language = ModelChoiceField(queryset=Language.objects.order_by('name'), required=True, label='Language', label_suffix='', to_field_name='code', widget=Select(attrs={'class': 'custom-select'}))

class SelectProbingTaskForm(Form):

    def __init__(self, language, *args, **kwargs):
        # Override method to filter ProbingTask by language parameter
        super().__init__(*args, **kwargs)
        self.fields['probing_task'] = ModelMultipleChoiceField(queryset=ProbingTask.objects.filter(languages__code=language).order_by('name'), required=True, label='Probing Tasks', label_suffix='', widget=CheckboxSelectMultiple(attrs={'class': 'form-check-input'}))

class UploadModelForm(Form):

    model = FileField(required=True, label='Model', label_suffix='', widget=FileInput(attrs={'class': 'custom-file-input'}))

    def clean_model(self):
        data = self.cleaned_data['model']
        _, extension = os.path.splitext(data.name)
        if extension not in ['.gz', '.vec']:
            raise ValidationError('File extension is invalid.')
        return data

class UploadEpochForm(Form):

    epoch = FileField(required=False, label='Epochs', label_suffix=' (Optional)', widget=FileInput(attrs={'class': 'custom-file-input', 'multiple': True}))

class SelectLayerForm(Form):

    def __init__(self, layer, *args, **kwargs):
        # Override method to add layer choices
        super().__init__(*args, **kwargs)
        self.fields['layer'] = ChoiceField(choices=[('', '---------')] + layer, required=True, label='Layer', label_suffix='', widget=Select(attrs={'class': 'custom-select'}))
