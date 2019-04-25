from allennlp.models.archival import load_archive
from allennlp.models.simple_tagger import SimpleTagger

from django.core.files.storage import FileSystemStorage

from .models import Model

def load_model(uuid):
    model = Model.objects.get(id=uuid)
    archive = load_archive(model.model.path)
    return archive.model

def get_modules(model):
    modules = list()
    if isinstance(model, SimpleTagger):
        encoder = model.encoder._module
        modules_as_string = [str(module) for module in encoder.modules()]
        modules = [(idx, module[:module.index('(')]) for idx, module in enumerate(modules_as_string)]
    else:
        raise NotImplementedError
    return modules
