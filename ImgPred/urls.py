from django.contrib import admin
from django.urls import path
from .views import Main

urlpatterns = [
    path('testyourimage/', Main.as_view(), name="testyourimage")
]
