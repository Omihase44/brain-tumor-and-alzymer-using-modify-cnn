from django.urls import path
from . import views
from .views_brain_tumor_v2 import (
    brain_tumor_classification_v2,
    brain_tumor_api_classify,
    brain_tumor_info
)


urlpatterns = [
    path('', views.home, name='home'),
    path('Admin_login/',views.Admin_login,name='Admin_login'),
    path('brain/', brain_tumor_classification_v2, name='brain'),
    path('Alzhimers/',views.Alzhimers,name='Alzhimers'),
    path('analysis/',views.analysis,name='analysis'),
    path('logout/',views.logout,name='logout'),
    path('View_Users/',views.View_Users,name='View_Users'),
    path('base/',views.base,name='base'),
    path('api/classify/', brain_tumor_api_classify, name='api_classify'),
    path('brain-info/', brain_tumor_info, name='brain_info'),


    ]

