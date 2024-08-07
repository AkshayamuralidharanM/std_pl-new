
from django.urls import path
from . import views

urlpatterns = [
    
    path('index',views.index),
    path('home',views.home),
    path('',views.login),
    path('logout',views.logout),
    path('datasetreg',views.datasetreg),
    path('predict',views.predict),
    path('modelpredict',views.modelpredict),
    path('delt',views.delt),
    path('studdelt',views.studdelt),
    path('update',views.update),
    path('mp_confirm',views.mp_confirm),
    path('hoddelt',views.hoddelt),
    path('tutdelt',views.tutdelt),
]
