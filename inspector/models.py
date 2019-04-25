from django.db.models import CharField, FileField, ManyToManyField, Model, UUIDField

from uuid import uuid4

# Create your models here.
class Language(Model):

    name = CharField(max_length=20, unique=True)
    code = CharField(max_length=2, unique=True)

    def __str__(self):
        return self.name

class ProbingTask(Model):

    name = CharField(max_length=20, unique=True)
    languages = ManyToManyField(Language)

    def __str__(self):
        return self.name

class Model(Model):

    id = UUIDField(primary_key=True, default=uuid4, editable=False)
    model = FileField(upload_to='models/')

    def __str__(self):
        return self.model.name
