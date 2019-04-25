from django.urls import path

from inspector.views import IndexView, SelectLanguageView, SelectLayerView, SelectProbingTaskView, UploadModelView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('language/', SelectLanguageView.as_view(), name='select_language'),
    path('language/probing-task/', SelectProbingTaskView.as_view(), name='select_probing_task'),
    path('language/probing-task/model/', UploadModelView.as_view(), name='upload_model'),
    path('language/probing-task/model/layer/', SelectLayerView.as_view(), name='select_layer')
]
