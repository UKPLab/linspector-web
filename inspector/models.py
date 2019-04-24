from django.db import models

# Create your models here.
class Language(models.Model):
    name = models.CharField(max_length=20, unique=True)
    code = models.CharField(max_length=2, unique=True)

    def __str__(self):
        return self.name

class ProbingTask(models.Model):
    name = models.CharField(max_length=20, unique=True)
    languages = models.ManyToManyField(Language)

    def __str__(self):
        return self.name
