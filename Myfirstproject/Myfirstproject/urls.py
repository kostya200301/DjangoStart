"""Myfirstproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('prediction/', views.Preds, name = 'Preds'),
    path('', views.home),
    path('yandex/', views.stock1, name='stock1'),
    path('microsoft/', views.stock2, name='stock2'),
    path('facebook/', views.stock3, name='stock3'),
    path('netflix/', views.stock4, name='stock4'),
    path('amazon/', views.stock5, name='stock5'),
]
