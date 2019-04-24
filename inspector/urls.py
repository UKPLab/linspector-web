from django.urls import path

from inspector.views import IndexView, SelectProbingTaskView, SelectLanguageView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('language/', SelectLanguageView.as_view(), name='select_language'),
    path('language/probing-task/', SelectProbingTaskView.as_view(), name='select_probing_task')
]
