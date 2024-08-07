from django.shortcuts import render

# Create your views here.
from django.shortcuts import render,redirect
from django.http import HttpResponse
from . models import User_mp
from  student.models import User_mp as mp
from student.models import stdworks_tbl
# from Master_Tutor_Registration.models import minorwork_tbl
from tutor.models import course
from django.http import HttpResponse
from django.conf import settings
from django.core.mail import send_mail
import os
from django.http.response import Http404
import random
# from  tutor.models import appre_tbl
import  datetime

def mipremail(request):
    if request.method=="POST":

        subject=request.POST.get("subjectmp")
        msg=request.POST.get("messagemp")
        to=request.POST.get("tomp")
        res=send_mail(subject,msg,settings.EMAIL_HOST_USER,[to])
        if(res==1):
            msg="mail send successfully"
            return render(request, 'student/mpemail.html', {'msg': msg})
        else:
            msg="mail could not send"
            return render(request, 'student/mpemail.html', {'msg': msg})
    else:
        msg=" "
        return render(request,'student/mpemail.html',{'msg':msg})

# Create your views here.

def loginpage(request):
    return render(request,'student/MinorLogin.html')

# Create your views
def home(request):
    user=request.session['username']
    passw=request.session['password']
    if user==""and passw=="":
        return redirect("student/loginpage")
    else:
        obj = mp.objects.filter(email=user, passwd=passw)
        for ls in obj:
            stdid=ls.id
        # ap = appre_tbl.objects.filter(studentid=stdid)

        return render(request,'student/minor_homepage.html',{'appre':'ap'})
def logout(request):
    request.session['username']=""
    request.session['password']=""
    return redirect("/student/loginpage")
def mpreg(request):
    if request.method=="POST":
        a = request.POST.get("mpname")
        b= request.POST.get("address")
        c= request.POST.get("phoneno")
        d=request.POST.get("dob")
        # f = request.POST.get("username")
        # g = request.POST.get("passwd")
        h = request.POST.get("email")
        i = request.POST.get("gender")
        j=request.POST.get("gn")
        k=request.POST.get("gpn")
        p=request.POST.get("mcs")
        q=request.POST.get("sname")
        rno=random.randrange(499,699)
        rno=str(rno)
        passw=a[0:3]+rno

        obj=mp.objects.create(mpname=a,address=b,phoneno=c,dob=d,username=h,passwd=passw,email=h,gender=i,gn=j,gpn=k,vrf="not",mcs=p,sname=q)
        obj.save()
        if obj:
            l = "successfully registered  username and password will send to mail after the confirmation"


            return render(request, 'student/mp_Registration.html', {"success": l})
        else:
            l = " not successfully registered"
            return render(request, 'student/mp_Registration.html', {"success": l})

    else :
            obj2 = course.objects.all()
            return render(request,'student/mp_Registration.html' ,{'data':obj2})



def edt(request):
        idno = request.POST.get("mpid")
        obj = User_mp.objects.filter(id=idno)
        adr=request.POST.get("addrs")
        ph=request.POST.get("ph")
        upd=User_mp.objects.get(id=idno)
        upd.address=adr
        upd.phoneno=ph
        upd.save()
        return redirect("/student/view_mp")

def update(request):
        if request.method == "POST":
            a = request.POST.get("mpname")
            b = request.POST.get("address")
            c = request.POST.get("phoneno")
            e = request.POST.get("dob")
            f = request.POST.get("username")
            g = request.POST.get("passwd")
            h = request.POST.get("email")

            j = request.POST.get("gn")
            k= request.POST.get("gpn")
            p= request.POST.get("mcs")
            q= request.POST.get("sname")

            idno = request.POST.get("mpid")
            upobj = User_mp.objects.get(id=idno)
            upobj.mpname = a
            upobj.address = b
            upobj.phoneno = c
            upobj.dob = e
            upobj.username = f
            upobj.passwd = g
            upobj.email = h

            upobj.gn= j
            upobj.gpn=k
            upobj.mcs=p
            upobj.sname=q
            upobj.save()
            return redirect("/tutorapp/home")
# def  view_mp(request):
#     if request.GET.get("idn"):
#         idno = request.GET.get("idn")
#         mpobj = mp.objects.get(id=idno)
        
    

    


def index(request):
    return render(request,'index.html')

def onlineeditor(request):
    return render(request,'onlineeditor.html')


def mpegis(request):
    trobj = User_mp.objects.all()
    frm=request.GET.get('frm')
    print(frm)
    if frm=='ADM':
        return render(request, 'adminApp/admin_homepage.html', {'data': trobj})
    else:
        return render(request, 'student/mp_details.html', {'data': trobj})


    
def mpegisT(request):
    if request.GET.get("msg"):
        msg=request.GET.get("msg")
        mpobj = User_mp.objects.all()
        return render(request, 'mpdetailT.html', {'data': mpobj,'msg':msg})

    mpobj=User_mp.objects.all()
    return render(request,'mpdetailT.html',{'data':mpobj})

def minorwork(request):
        if request.method == "POST":
            assname= request.POST.get("an")
            assign= request.FILES.get("assignment")
            # assdate= request.POST.get("adt")
            stdcourse = request.POST.get("stdcrs")
            stdname = request.POST.get("stdname")
            obj = stdworks_tbl.objects.create(assname=assname, assign=assign, stdcourse=stdcourse, stdname=stdname)
            obj.save()
            if obj:
                t = "successfully registered"

                return render(request, 'student/minoruploadwork.html', {"success": t})
            else:
                t = " not successfully registered"
                return render(request, 'student/minoruploadwork.html', {"success": t})

        else:
            user = request.session['username']
            passw = request.session['password']
            if user == "" and passw == "":
                return redirect("/student/")
            else:
                mobj=mp.objects.filter(username=user,passwd=passw)
                return render(request, 'student/minoruploadwork.html', {"data": mobj})







def viewminorwork(request):
    vmwobj = minorwork_tbl.objects.all()


    return render(request, 'miass.html', {'data': vmwobj})

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
        obj=mp.objects.filter(email=username,passwd=password)
        if obj:
            request.session['username']=username
            request.session['password']=password
            for l in obj:
                idno = l.id
            request.session['idno']=idno
            return redirect("/student/home")
        else:
            request.session['username']=""
            request.session['password']=""
            msg="username or password incorrect"
            
            # ap = appre_tbl.objects.filter(studentid=stdid)
def profile(request):
    idn = request.session['idno']
    obj = mp.objects.filter(id=idn)
    return render(request,'student/viewprofile.html',{'stud':obj})

            
def shareprofile(request):
    obj = User_mp.objects.filter(username='appu123', passwd='1234adithya')
    msg = ""
    return render(request, 'shareprofile.html', {'data': obj, 'msg': msg})