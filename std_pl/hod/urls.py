from django.contrib import admin
from django.urls import path
from . import views

urlpatterns =[

    path("hod_reg",views.hod_reg),
    path('home',views.home),
    path('login',views.login),
    path('hod_details',views.hregis),
    path('hodmail',views.hodmail),
    path('uploadnotes',views.uploadnotes),
    path('uploadworks',views.uploadworks),
    path('logout',views.logout),
    path('seeassignh',views.seeassignh),
]