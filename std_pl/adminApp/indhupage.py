from django.shortcuts import render,redirect
from django.http import HttpResponse
from tutorapp.models import tutorReg_tbl
from Minor_Programmer.models import User_mp
from Master_Tutor_Registration.models import  User_mtr
from . models import notification_tbl
from django.http import HttpResponse
from django.conf import settings
from django.core.mail import send_mail
# MatPlotLib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

def mail(request):
    if request.method == "POST":
      subject = request.POST.get("subjectad")
      msg=request.POST.get("messagead")
      to = request.POST.get("admail")
      res=send_mail(subject,msg,settings.EMAIL_HOST_USER,[to])
      if(res==1):
         msg="mail send successfully"
         return render(request, 'adminemail.html', {'msg': msg})
      else:
         msg = "mail could not send"
         return render(request, 'adminemail.html', {'msg': msg})
    else:
         msg = " "
         return render(request, 'adminemail.html', {'msg': msg})
def regnewcourse(request):
    return render(request,'Regnewcourse.html')

# views Login page

def loginpage(request):
    return render(request,'AdminLogin.html')
#Home    
def home(request):
    user=request.session['username']
    passw=request.session['password']
    if user==""and passw=="":
        return redirect("/admin_module/")
    else:
        return render(request,'admin_homepage.html')


    
    
def notification(request):
    if request.method=="POST":
        x = request.POST.get("nt")
        y= request.POST.get("nd")
        z= request.POST.get("message")

        obj=notification_tbl.objects.create(notititle=x,notidate=y,msg=z)
        obj.save()
        if obj:
            q = "successfully registered"
            return render(request, 'notification.html', {"success": q})
        else:
            q = " not successfully registered"
            return render(request, 'notification.html', {"success": q})

    else :
            return render(request,'notification.html')


def viewnots(request):
    nobj = notification_tbl.objects.all()

    return render(request, 'viewnoti.html', {'data': nobj})
def tutnots(request):
    tnonj=notification_tbl.objects.all()
    return render(request,'tutnoti.html',{'data':tnonj})
def mastnots(request):
    maonj=notification_tbl.objects.all()
    return render(request,'masternoti.html',{'data':maonj})
def piechart():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Tutor', 'MinorProgrammer'
    mastr=User_mtr.objects.all().count()
    totr = tutorReg_tbl.objects.all().count()
    mpr = User_mp.objects.all().count()
    sizes = [mastr,totr, mpr]
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('Assets/Mst.png',dpi=100)
#login checking     
def login(request):
    if request.method=="POST":
        username=request.POST.get("username")
        password=request.POST.get("password")
        obj=admin_tbl.objects.filter(username=username,password=password)
        if obj:
            request.session['username']=username
            request.session['password']=password
            return redirect("/admin_module/home")
        else:
            request.session['username']=""
            request.session['password']=""

            return render(request,'AdminLogin.html',{'lmsg':msg})
    else:
        msg=" "
        return render(request,'AdminLogin.html',{'lmsg':msg})

def mp_confirm(request):
    idno = request.GET.get("idn")
    mpobj = User_mp.objects.get(id=idno)
    mpobj.vrf = "confirm"
    mpobj.save()
    return redirect("/home")