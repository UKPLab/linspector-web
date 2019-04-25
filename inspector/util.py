from django.core.exceptions import ObjectDoesNotExist, SuspiciousOperation
from django.core.files.storage import FileSystemStorage

from .models import Language, Model, ProbingTask

def get_request_params(request):
    if 'lang' in request.GET:
        try:
            language = Language.objects.get(code=request.GET['lang'])
            if 'task' in request.GET:
                try:
                    split = request.GET['task'].split(',')
                    probing_tasks = ProbingTask.objects.filter(pk__in=split, languages__code=language.code)
                    if len(split) != len(probing_tasks):
                        raise ObjectDoesNotExist()
                    if 'model' in request.GET:
                        try:
                            model = Model.objects.get(id=request.GET['model'])
                            return (language, probing_tasks, model)
                        except ObjectDoesNotExist:
                            raise SuspiciousOperation('`model` parameter is invalid.')
                    else:
                        return (language, probing_tasks)
                except (ObjectDoesNotExist, ValueError):
                    raise SuspiciousOperation('`task` parameter is invalid.')
            else:
                return (language)
        except ObjectDoesNotExist:
            raise SuspiciousOperation('`lang` parameter is invalid.')
    else:
        return None
