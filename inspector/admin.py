from django.contrib import admin

from .models import Language, ProbingTask

# Register your models here.
admin.site.register(Language)
admin.site.register(ProbingTask)
