# recorder/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start/', views.start_recording, name='start_recording'),
    path('stop/', views.stop_recording, name='stop_recording'),
]
