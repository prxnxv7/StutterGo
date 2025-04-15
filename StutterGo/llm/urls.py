from django.urls import path
from .views import stutter_analysis_view

urlpatterns = [
    path('analyze/',stutter_analysis_view, name='analyze-stutter'),
]
