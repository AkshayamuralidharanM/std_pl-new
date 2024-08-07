from django.shortcuts import render

# Create your views here.
from django.shortcuts import render,redirect
from django.http import HttpResponse

from  student.models import User_mp as mp
# from Master_Tutor_Registration.models import minorwork_tbl
from tutor.models import course
from django.http import HttpResponse
from django.conf import settings
from django.core.mail import send_mail
from django.shortcuts import render,redirect
from django.http import HttpResponse
from tutor.models import tutorReg_tbl

from hod.models import hodReg_tbl
from django.http import HttpResponse
from django.conf import settings
from django.core.mail import send_mail
import random
from django.conf import settings
from django.core.mail import send_mail
from hod.models import Notes_tbl,Works_tbl
from student.models import stdworks_tbl

# MatPlotLib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

def password_gen():
    sp=['@','#','$','%','&','*',')']
    
    print(ord('A'),ord('Z'),ord('a'),ord('z'))
    
    codeno=""

    for i in range(0,2):
        j=random.randint(0,6)
        
        codeno=codeno+chr(random.randrange(65,90))+sp[j]+str(random.randint(0,9))
    return codeno       

def hodmail(request):
    if request.method=="POST":

        subject=request.POST.get("subjecttut")
        msg=request.POST.get("messagetut")
        to=request.POST.get("totut")
        res=send_mail(subject,msg,settings.EMAIL_HOST_USER,[to])
        if(res==1):
            msg="mail send successfully"
            return render(request,'hod/hodmail.html',{'msg': msg})
        else:
            msg="mail could not send"
            return render(request,'hod/hodmail.html',{'msg': msg})
    else:
        msg=" "
        return render(request,'hod/hodmail.html',{'msg':msg})
# Create your views here.
#def home(request):
    #return HttpResponse("hello")
def home(request):
    return render(request,'hod/hod_homepage.html')
def login(request):
    if request.method=="POST":
        username=request.POST.get("user")
        password=request.POST.get("passw")
        print(username)
        obj=hodReg_tbl.objects.filter(username=username,passwd=password)
        if obj:
            request.session['username']=username
            request.session['password']=password
            return redirect("/hod/home")
            # return render(request,'tutor/tutor_homepage.html')
        else:
            request.session['username']=""
            request.session['password']=""

            return render(request,'hod/hod_login.html',{'lmsg':"check your data"})
    else:
        msg=" "
        return render(request,'hod/hod_login.html',{'lmsg':msg})

def hod_reg(request):
    if request.method == "POST":
        a = request.POST.get("tname")
        b = request.POST.get("add")
        c = request.POST.get("mb")
        e = request.POST.get("dpt")
        f = request.POST.get("user")
        g = request.POST.get("pss")
        h = request.POST.get("eml")
        i = request.POST.get("gen")
       # j = request.POST.get("desi")
        pic=request.FILES.get('timage')
        print(a,b)
        rno = random.randrange(499, 699)
        rno = str(rno)
        passw = a[0:3] + rno


        obj=hodReg_tbl.objects.create(hname=a, address=b, phoneno=c, dpt="dpt", username=f, passwd=g,email=h, gender=i, desi="des",photo=pic)
        obj.save()
        if obj:
            subject = "Username and Password"
            msg = "Your Username:" + h + "\n Password:" + passw + "\n Login using this link http://127.0.0.1:8000/hod/"
            to = h
            # res = send_mail(subject, msg, settings.EMAIL_HOST_USER, [to])
            # if res:
            #     l = "successfully registered mail send"
            # else:
            #     l="registered  successfully mail not send"

            return render(request, 'hod/hod_login.html', {"success": "successfully registered mail send",'user':h,'password':g})
        else:
            # l = " not successfully registered"
            return render(request, 'hod/hod_login.html', {"success":" Not successfully registered mail send"})

    else:
        return render(request, 'hod/hod_reg.html')
def notes(request):
    return render(request,'notes.html')
def loginpage(request):
    return render(request,'hod_login.html')

# Create your views
def home(request):
    user=request.session['username']
    passw=request.session['password']
    if user==""and passw=="":
        return redirect("/hod/")
    else:

        return render(request,'hod/hod_homepage.html')
def logout(request):
    request.session['username']=""
    request.session['password']=""
    return redirect("/hod/login")



def index(request):
    return render(request,'hod/hod_login.html')




def tregis(request):
    trobj = tut.objects.all()
    frm=request.GET.get('frm')
    print(frm)
    if frm=='ADM':
        return render(request, 'admintutor.html', {'data': trobj})
    else:
        return render(request, 'tutordetails.html', {'data': trobj})

    # return render(request, 'tutordetails.html', {'data': trobj})
    #


def edt(request):
    idno = request.GET.get("tutid")
    obj = tut.objects.filter(id=idno)
    return render(request, 'edit.html', {'data': obj})


def update(request):
    if request.method == "POST":
        a = request.POST.get("tname")
        b = request.POST.get("address")
        c = request.POST.get("phoneno")
        e = request.POST.get("dpt")
        f = request.POST.get("username")
        g = request.POST.get("passwd")
        h = request.POST.get("email")
        i = request.POST.get("gender")
        j = request.POST.get("desi")
        idno = request.POST.get("tutid")
        upobj = tut.objects.get(id=idno)
        upobj.tname = a
        upobj.address = b
        upobj.phoneno = c
        upobj.dpt = e
        upobj.username = f
        upobj.passwd = g
        upobj.email = h
        upobj.gender = i
        upobj.desi = j
        upobj.save()
        return redirect("/tutorapp/tutdetails")


def uploadnotes(request):
    if request.method=="POST":
        title=request.POST.get("tt")
        note=request.FILES.get("notes")
        date=request.POST.get("dt")
        course1=request.POST.get("course1")
        standard=request.POST.get("std")
        obj = Notes_tbl.objects.create(title=title, url=note, dt=date, crs=course1, status=standard)
        obj.save()
        if obj:
            s = "successfully registered"

            return render(request, 'hod/uploadnotesT.html', {"success": s})
        else:
            s = " not successfully registered"
            return render(request, 'hod/uploadnotesT.html', {"success": s})

    else:
        crs = course.objects.all()

        return  render(request,'hod/uploadnotesT.html',{'data':crs})

def uploadworks(request):
        if request.method == "POST":
            wtitle = request.POST.get("wt")
            work = request.FILES.get("works")
            wdate = request.POST.get("wdt")
            course1 = request.POST.get("course")
            stand = request.POST.get("std")
            obj = Works_tbl.objects.create(wt=wtitle, notes=work, wdt=wdate, course=course1, std=stand)
            obj.save()
            if obj:
                t = "successfully registered"

                return render(request, 'hod/uploadworkT.html', {"success": t})
            else:
                t = " not successfully registered"
                return render(request, 'hod/uploadworkT.html', {"success": t})

        else:
            #obj= mp.objects.all()
            crs=course.objects.all()
            return render(request, 'hod/uploadworkT.html',{"data":crs})

def Notes(request):
            noobj = Notes_tbl.objects.all()
            return render(request, 'notes.html', {'data': noobj})


def Works(request):
    wobj = Works_tbl.objects.all()
    return render(request, 'works.html', {'data': wobj})

def download_file(request):
        # fill these variables with real values
        fl = request.GET.get('filename')
        file_path = os.path.join(settings.MEDIA_ROOT, fl)
        print(file_path)
        # pos=fl.index(".")
        # fnl=fl[0,pos+1]
        # filename =fnl+'.xls'

        # fl=open(fl_path,"r")
        # mime_type,_= mimetypes.guess_type(fl_path)
        # response = HttpResponse(fl, content_type=mime_type)
        # response['Content-Disposition'] = "attachment; filename=%s" % filename
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type='application/pdf')
                response['content-Disposition'] = 'inline;filname=' + os.path.basename(file_path)
                return response
        raise Http404



def login(request):
    if request.method=="POST":
        username=request.POST.get("user")
        password=request.POST.get("passw")
        obj=hodReg_tbl.objects.filter(username=username,passwd=password)
        if obj:
            request.session['username']=username
            request.session['password']=password
            return redirect("/hod/home")
            # return render(request,'tutor/tutor_homepage.html')
        else:
            request.session['username']=""
            request.session['password']=""

            return render(request,'hod/hod_login.html',{'lmsg':"check your data"})
    else:
        msg=" "
        return render(request,'hod/hod_login.html',{'lmsg':msg})
def appre(request):
    stid=request.GET.get("sdtid")
    colr=request.GET.get("colr")
    mp1=mp.objects.filter(id=stid)
    for ls  in mp1:
        stdname=ls.mpname
    chck = appre_tbl.objects.filter(studname=stdname, colour=colr)
    if chck:
        msg = "Already send"
        return redirect("/Minor_Programmer/mpdetailsT?msg=Already send")


    if colr=="red":
        url="images/badge.png"
    if colr=="green":
        url="images/ribbon.png"
    if colr=="blue":
        url="images/best-seller.png"

    frkid=mp.objects.get(id=stid)
    print(frkid)
    obj=appre_tbl.objects.create(studentid=frkid,studname=stdname,appreciation=url,colour=colr)
    if obj:
        obj.save()
        msg="Appreciation send"
        return redirect("/Minor_Programmer/mpdetailsT?msg=Appreciation send")
    else:
        msg = "Error in Sending Appreciation "
        return redirect("/Minor_Programmer/mpdetailsT?msg=Error in Sending Appreciation")
def delt(request):
    idno=request.GET.get("idn")
    mast=tut.objects.get(id=idno)
    mast.delete()
    return redirect("/tutorapp/tutdetails")

def hregis(request):
    trobj = hodReg_tbl.objects.all()
    frm=request.GET.get('frm')
    print(frm)
    if frm=='ADM':
        return render(request, 'adminApp/admin_homepage.html', {'data': trobj})
    else:
        return render(request, 'hod/hod_details.html', {'data': trobj})
def seeassignh(request):
    if request.method=="POST": 
        crs=request.POST.get("course1")
        dat=stdworks_tbl.objects.all()
        crs1=course.objects.all()
        return render(request,'hod/seeassignh.html',{'data':dat,'crs':crs1})
    else:
        crs=course.objects.all()
        return render(request, 'hod/seeassignh.html',{"crs":crs}) 

       