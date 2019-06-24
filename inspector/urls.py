from django.urls import path

from inspector.views import AboutView, IndexView, ProbeView, SelectLanguageView, SelectLayerView, SelectProbingTaskView, ShowResultView, UploadEpochView, UploadModelView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('about/', AboutView.as_view(), name='about'),
    path('language/', SelectLanguageView.as_view(), name='select_language'),
    path('language/probing-task/', SelectProbingTaskView.as_view(), name='select_probing_task'),
    path('language/probing-task/model/', UploadModelView.as_view(), name='upload_model'),
    path('language/probing-task/model/epoch/', UploadEpochView.as_view(), name='upload_epoch'),
    path('language/probing-task/model/epoch/layer/', SelectLayerView.as_view(), name='select_layer'),
    path('language/probing-task/model/probe/', ProbeView.as_view(), name='probe_view'),
    path('language/probing-task/model/epoch/layer/probe/', ProbeView.as_view(), name='probe_view'),
    path('language/probing-task/model/probe/result/', ShowResultView.as_view(), name='show_result'),
    path('language/probing-task/model/epoch/layer/probe/result/', ShowResultView.as_view(), name='show_result'),
    path('<uuid:id>/', ShowResultView.as_view(), name='show_result')
]
