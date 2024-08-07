from django.urls import path
from . import views
urlpatterns=[
    path('tutor_reg',views.tureg),
    path('singin',views.login),
    path('logout',views.logout),
    path('home',views.home),
    path('',views.index),

    path("tutdetails", views.tregis),
    path("edit", views.edt),
    path('update', views.update),
    path('uploadnotes', views.uploadnotes),
    path('uploadworks',views.uploadworks),
    path('Notes',views.Notes),
    path('Works',views.Works),
    path("tutemail",views.tutemail),
    path('download_file',views.download_file),
    path('seeassignt',views.seeassignt),
    path('datasetreg',views.datasetreg),
    path('predict',views.predict),
    path('modelpredict',views.modelpredict),
     
   
    path('logout',views.logout),
    path('appre',views.appre),
    path('delt',views.delt),
]