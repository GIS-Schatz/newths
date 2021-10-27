from django.contrib import admin
from django.core.management import templates
from django.urls import path, include
from newths import views

urlpatterns = [
    path('', views.index, name='index'),
    path('admin/', admin.site.urls),
    path('beerAI1/', views.ver1, name='ver1'),
    path('beerAI2/', views.ver2, name='ver2'),
]