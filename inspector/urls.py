from django.contrib.sitemaps.views import sitemap
from django.urls import path

from .sitemaps import StaticViewSitemap
from .views import AboutView, IndexView, PaperView, ProbeView, SelectLanguageView, SelectLayerView, SelectProbingTaskView, ShowResultView, UploadEpochView, UploadModelView

sitemaps = {'static': StaticViewSitemap}

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('about/', AboutView.as_view(), name='about'),
    path('paper/', PaperView.as_view(), name='paper'),
    path('language/', SelectLanguageView.as_view(), name='select_language'),
    path('language/probing-task/', SelectProbingTaskView.as_view(), name='select_probing_task'),
    path('language/probing-task/model/', UploadModelView.as_view(), name='upload_model'),
    path('language/probing-task/model/epoch/', UploadEpochView.as_view(), name='upload_epoch'),
    path('language/probing-task/model/epoch/layer/', SelectLayerView.as_view(), name='select_layer'),
    path('language/probing-task/model/probe/', ProbeView.as_view(), name='probe_view'),
    path('language/probing-task/model/epoch/layer/probe/', ProbeView.as_view(), name='probe_view'),
    path('language/probing-task/model/probe/result/', ShowResultView.as_view(), name='show_result'),
    path('language/probing-task/model/epoch/layer/probe/result/', ShowResultView.as_view(), name='show_result'),
    path('<uuid:id>/', ShowResultView.as_view(), name='show_result'),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}, name='django.contrib.sitemaps.views.sitemap')
]
