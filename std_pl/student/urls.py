from django.contrib import admin
from django.urls import path
from . import views

urlpatterns =[

    path("login",views.login),
    path("mpreg",views.mpreg),
    path("",views.index),
    path("mpdetails",views.mpegis),
    path("mpdetailsT",views.mpegisT),
    path("minorwork",views.minorwork),
    path("viewminorwork",views.viewminorwork),
    path("mipremail",views.mipremail),
    path('download_file',views.download_file),
    path("edit", views.edt),
    path('update', views.update),
    path('loginpage',views.loginpage),
    path('home',views.home),
    path('logout',views.logout),
    path('profile',views.profile),
    path('onlineeditor',views.onlineeditor),
    path('shareprofile',views.shareprofile),
    path('download_file',views.download_file),
       


]
