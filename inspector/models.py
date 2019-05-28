from django.db.models import BooleanField, CharField, FileField, ManyToManyField, Model, UUIDField
from django.db.models.signals import pre_delete
from django.dispatch import receiver

import os

from uuid import uuid4

class Language(Model):

    name = CharField(max_length=20, unique=True)
    code = CharField(max_length=2, unique=True)

    def __str__(self):
        return self.name

class ProbingTask(Model):

    name = CharField(max_length=35, unique=True)
    languages = ManyToManyField(Language)
    contrastive = BooleanField(default=False)

    def __str__(self):
        return self.name

    def to_camel_case(self):
        return self.name.replace(' ', '')

class Model(Model):

    id = UUIDField(primary_key=True, default=uuid4, editable=False)
    upload = FileField()

    def __str__(self):
        return self.upload.name

@receiver(pre_delete, sender=Model)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """Auto delete file when model is deleted."""
    if instance.upload and os.path.isfile(instance.upload.path):
        os.remove(instance.upload.path)
