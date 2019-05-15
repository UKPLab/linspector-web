from celery import Celery

import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'linspector.settings')

app = Celery('linspector', broker='amqp://', backend='rpc://')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
