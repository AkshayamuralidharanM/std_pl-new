from adminApp.models import *
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render,redirect
from  django.http import HttpResponse
from tutor.models import tutorReg_tbl,Notes_tbl,Works_tbl,course
from django.http import HttpResponse
from django.conf import settings

import os
from django.http.response import Http404
import random
from django.conf import settings
from django.core.mail import send_mail
from student.models import stdworks_tbl,User_mp
import  numpy as np
import pickle

def tutemail(request):
    if request.method=="POST":

        subject=request.POST.get("sub")
        msg=request.POST.get("msg")
        to=request.POST.get("to")
        res=send_mail(subject,msg,settings.EMAIL_HOST_USER,[to])
        if(res==1):
            msg="mail send successfully"
            return render(request,'tutor/temail.html',{'msg': msg})
        else:
            msg="mail could not send"
            return render(request,'tutor/temail.html',{'msg': msg})
    else:
        msg=" "
        return render(request,'tutor/temail.html',{'msg':msg})
# Create your views here.
#def home(request):
    #return HttpResponse("hello")
def home(request):
    return render(request,'tutor/tutor_homepage.html')
def log_in(request):
    if request.method=="POST":
        usr=request.POST.get('user')
        pssw=request.POST.get('passw')
        obj=tutorReg_tbl.objects.filter(username=usr,passwd=pssw)
        if obj:
            return HttpResponse("Homepage")   
def tureg(request):
    if request.method == "POST":
        a = request.POST.get("tname")
        b = request.POST.get("add")
        c = request.POST.get("mb")
        e = request.POST.get("dpt")
        f = request.POST.get("use")
        g = request.POST.get("pss")
        h = request.POST.get("eml")
        i = request.POST.get("gen")
       # j = request.POST.get("desi")
        pic=request.FILES.get('timage')
        print(a,b)
        rno = random.randrange(499, 699)
        rno = str(rno)
        passw = a[0:3] + rno

        obj = tutorReg_tbl.objects.create(tname=a, address=b, phoneno=c, dpt="dpt", username=h, passwd=g,
                                     email=h, gender=i, desi="des",photo=pic)
        obj.save()
        if obj:
            subject = "Username and Password"
            msg = "Your Username:" + h + "\n Password:" + passw + "\n Login using this link http://127.0.0.1:8000/tutorapp/"
            to = h
            # res = send_mail(subject, msg, settings.EMAIL_HOST_USER, [to])
            # if res:
            #     l = "successfully registered mail send"
            # else:
            #     l="registered  successfully mail not send"

            return render(request, 'tutor/signin.html', {"success": "successfully registered mail send",'user':h,'password':g})
        else:
            # l = " not successfully registered"
            return render(request, 'tutor/Treg.html', {"success":" Not successfully registered mail send"})

    else:
        return render(request, 'tutor/Treg.html')
def notes(request):
    return render(request,'notes.html')
def loginpage(request):
    return render(request,'TutorLogin.html')

# Create your views
def home(request):
    user=request.session['username']
    passw=request.session['password']
    if user==""and passw=="":
        return redirect("/tutor/")
    else:

        return render(request,'tutor/tutor_homepage.html')
def logout(request):
    request.session['username']=""
    request.session['password']=""
    return redirect("/tutor/singin")



def index(request):
    return render(request, 'tutor/signin.html')


def tregis(request):
    trobj = tutorReg_tbl.objects.all()
    frm=request.GET.get('frm')
    print(frm)
    if frm=='ADM':
        return render(request, 'adminApp/admin_homepage.html', {'data': trobj})
    else:
        return render(request, 'tutor/tutor_details.html', {'data': trobj})

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
        #date=request.POST.get("dt")
        course1=request.POST.get("course1")
        standard=request.POST.get("std")
        obj = Notes_tbl.objects.create(title=title, url=note, crs=course1, status=standard)
        obj.save()
        if obj:
            s = "successfully registered"

            return render(request, 'tutor/uploadnotesT.html', {"success": s})
        else:
            s = " not successfully registered"
            return render(request, 'tutor/uploadnotesT.html', {"success": s})

    else:
        crs = course.objects.all()
        

        return  render(request,'tutor/uploadnotesT.html',{'data':crs})

def uploadworks(request):
        if request.method == "POST":
            wtitle = request.POST.get("wt")
            work = request.FILES.get("works")
            # wdate = request.POST.get("wdt")
            course1 = request.POST.get("course")
            stand = request.POST.get("std")
            obj = Works_tbl.objects.create(wt=wtitle, notes=work,course=course1, std=stand)
            obj.save()
            if obj:
                t = "successfully registered"

                return render(request, 'tutor/uploadworkT.html', {"success": t})
            else:
                t = " not successfully registered"
                return render(request, 'tutor/uploadworkT.html', {"success": t})

        else:
            obj= User_mp.objects.all()
            crs=course.objects.all()
            return render(request, 'tutor/uploadworkT.html',{"data":crs,'data1':obj})

def Notes(request):
            noobj = Notes_tbl.objects.all()
            return render(request, 'student/notes.html', {'data': noobj})

#********************************************************************pred********************************************************8

def datasetreg(request):
    if request.method=="POST":
        sname=request.POST.get('snam')
        gender=request.POST.get('g')
        ssc_p=request.POST.get('ssc_p')
        ssc_b=request.POST.get('ssc_b')
        hsc_p=request.POST.get('hsc_p')
        hsc_b=request.POST.get('hsc_b')
        hsc_s=request.POST.get('hsc_s')
        degree_p=request.POST.get('degree_p')
        degree_t=request.POST.get('degree_t')
        workex=request.POST.get('workex')
        etest_p=request.POST.get('etest_p')
        specialisation=request.POST.get('specialisation')
        mba_p=request.POST.get('mba_p')
        obj=Dataset_tbl.objects.create(sname=sname,
        g=gender,ssc_p=ssc_p,ssc_b=ssc_b,hsc_p=hsc_p,hsc_b=hsc_b,hsc_s=hsc_s,degree_p=degree_p,degree_t=degree_t,workex=workex,etest_p=etest_p,specialisation=specialisation,mba_p=mba_p,salary=000)
        obj.save()
        if obj:
            msg="Successfully"
            obj=Admin_tbl.objects.all()
            data=Dataset_tbl.objects.all()
            return render(request,'home_t.html',{'data':obj,'dataset':data})
            
        else:
            msg="Not Successfully"
            return render(request,'Dataset_reg_t.html',{'msg':msg})    
    msg=" "        
    return render(request,'Dataset_reg_t.html',{'msg':msg})



#prediction

def predict(request):
    idno=1
    testdata=[]
    obj=Dataset_tbl.objects.filter(id=idno) 
    for ls in obj:

        testdata.append(ls.g)
        testdata.append(ls.ssc_p)
        testdata.append(ls.ssc_b)
        testdata.append(ls.hsc_p)
        testdata.append(ls.hsc_b)
        testdata.append(ls.hsc_s)
        testdata.append(ls.degree_p)
        testdata.append(ls.degree_t)
        testdata.append(ls.workex)
        testdata.append(ls.etest_p)
        testdata.append(ls.mba_p)
        testdata.append(ls.specialisation)
        
        
        
    # for l in testdata:
        

    #     print(l,end=" ")
        # i+=1
    if testdata[0]=="Female":
        testdata[0]=0 
    if testdata[0]=="Male":
        testdata[0]=1 
    if testdata[0]=="others":
        testdata[0]=2  
    if testdata[2]=="Kerala":
        testdata[2]=0 
    if testdata[2]=="Central":
        testdata[2]=1 
    if testdata[2]=="Others":
        testdata[2]=2 
    if testdata[4]=="Kerala":
        testdata[4]=0 
    if testdata[4]=="Central":
        testdata[4]=1 
    if testdata[4]=="Others":
        testdata[4]=2
    if testdata[5]!="":
        testdata[5]=0 
    if testdata[7]!="":
        testdata[7]=1 
    if testdata[8]=="no":
        testdata[8]=0
    if testdata[8]=="yes":
        testdata[8]=1
    if testdata[11]!="":
        testdata[11]=0
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import numpy as np 
    import pickle


    dataset = pd.read_csv('dataset/Placement_Data_Full_Class.csv')
    print(dataset.head())
   
    dataset = dataset.drop('salary', axis=1)
    dataset = dataset.drop('sl_no', axis=1)
    features_to_split = ['hsc_s','degree_t']
    
    # catgorising col for further labelling
    dataset["gender"] = dataset["gender"].astype('category')
    dataset["ssc_b"] = dataset["ssc_b"].astype('category')
    dataset["hsc_b"] = dataset["hsc_b"].astype('category')
    dataset["degree_t"] = dataset["degree_t"].astype('category')
    dataset["workex"] = dataset["workex"].astype('category')
    dataset["specialisation"] = dataset["specialisation"].astype('category')
    dataset["status"] = dataset["status"].astype('category')
    dataset["hsc_s"] = dataset["hsc_s"].astype('category')
    print(dataset.dtypes)
    
    # labelling the columns
    dataset["gender"] = dataset["gender"].cat.codes
    dataset["ssc_b"] = dataset["ssc_b"].cat.codes
    dataset["hsc_b"] = dataset["hsc_b"].cat.codes
    dataset["degree_t"] = dataset["degree_t"].cat.codes
    dataset["workex"] = dataset["workex"].cat.codes
    dataset["specialisation"] = dataset["specialisation"].cat.codes
    dataset["status"] = dataset["status"].cat.codes
    dataset["hsc_s"] = dataset["hsc_s"].cat.codes
    
    # display dataset
    print(dataset)
   

    # selecting the features and labels
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    # display dependent variables
    print("y",Y)

    # dividing the data into train and test

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)

# display dataset
    print(dataset.head())

    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    #save model
    pkl_filename = "LR_model.pkl"
    pkl_gnb="GNB_model.pkl"

    with open(pkl_filename, 'wb') as file:
         pickle.dump(lr, file)


     

   


    y_pred = lr.predict([[0,74.00,2,59.00,2,1,73.00,1,0,60,0,56.7]])
    prt=lr.predict_proba(X_test)
    nbclassifier = GaussianNB()
    nbclassifier.fit(X_train, Y_train)
    y_pred_nb = nbclassifier.predict([testdata])
    accuracy_score(Y_test, y_pred_nb)
    nbclassifier.score(X_train, Y_train)
    filename = 'GNB_model.sav'
    pickle.dump(nbclassifier, open(filename, 'wb'))   
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, Y_test)
    print(X_test)
    print(y_pred_nb)
    prt=np.argmax(prt)
    print(prt*100,"%")

    for p in y_pred_nb:
        if p==1:
            print("Placed")
        else:
            print("NotPlaced")   
    #knn
    from sklearn.neighbors import KNeighborsClassifier as KNN
    knn = KNN(n_neighbors = 3)
    
    # train model
    knn.fit(X_train, Y_train)

    # printing the acc
    print(knn.score(X_test, Y_test))

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(knn, open(filename, 'wb'))
    print("Predict knn",knn.predict([testdata]))
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, Y_test)
    print("Knn",result)        



    return redirect("/home")  

def modelpredict(request):
    idno=request.GET.get('idn')
    testdata=[]
    obj=Dataset_tbl.objects.filter(id=idno) 
    i=0
    for ls in obj:

        testdata.append(ls.g)
        testdata.append(ls.ssc_p)
        testdata.append(ls.ssc_b)
        testdata.append(ls.hsc_p)
        testdata.append(ls.hsc_b)
        testdata.append(ls.hsc_s)
        testdata.append(ls.degree_p)
        testdata.append(ls.degree_t)
        testdata.append(ls.workex)
        testdata.append(ls.etest_p)
        testdata.append(ls.mba_p)
        testdata.append(ls.specialisation)
        
        
    # for l in testdata:
        

    #     print(l,end=" ")
        # i+=1
    if testdata[0]=="Female":
        testdata[0]=0 
    if testdata[0]=="Male":
        testdata[0]=1 
    if testdata[0]=="others":
        testdata[0]=2  
    if testdata[2]=="Kerala":
        testdata[2]=0 
    if testdata[2]=="Central":
        testdata[2]=1 
    if testdata[2]=="Others":
        testdata[2]=2 
    if testdata[4]=="Kerala":
        testdata[4]=0 
    if testdata[4]=="Central":
        testdata[4]=1 
    if testdata[4]=="Others":
        testdata[4]=2
    if testdata[5]!="":
        testdata[5]=0 
    if testdata[7]!="":
        testdata[7]=1 
    if testdata[8]=="no":
        testdata[8]=0
    if testdata[8]=="yes":
        testdata[8]=1
    if testdata[11]!="":
        testdata[11]=0

    print(testdata)  
    df=np.array(testdata)
    df=df.astype(int)





             
    
   
    pkl_filename='D:\project\student_project\std_pl-new\std_pl\LR_model.pkl'
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
  
 
    
    # # Calculate the accuracy score and predict target values
    # # score = pickle_model.score(Xtest, Ytest)
    # # print("Test score: {0:.2f} %".format(100 * score))
    # vectors=TfidfVectorizer()
    # # labelencounter=LabelEncoder()
    # # ln=len(testdata)
    # for l in testdata:
    #     print(l,end=" ")




   

    # testdata=pd.DataFrame(testdata)
    # testdata=vectors.fit_transform([df])
    # testdata=vectors.transform([df]) [0,74.00,2,59.00,2,1,73.00,1,0,60,0,56.7]
    Ypredict = pickle_model.predict([df])

    prd=np.argmax(Ypredict)
    prec=pickle_model.predict_proba([df])
    pr=np.argmax(prd)
    pr=pr*100
    if prd==0:
        res="Placed"
        print("Placed")
    else:
        res="Not Placed  "+str(pr)+'%'
        print("Note placed")    
    print("Predict",Ypredict)
    print("predict",pr)
    #knn
    filename1="finalized_model.sav"
    loaded_model = pickle.load(open(filename1, 'rb'))
    result = loaded_model.predict([df])
    percent=loaded_model.predict_proba([df])
    print("knn percent",percent)
    percnt=np.argmax(percent[0][0]*100)
    percnt=(percent[0][1]*100).round(2)
    
    percnt=str(percnt)
    print("Knn",result)  
    rs=['Placed',"Not Placed"]
    prdknn=np.argmax(result)
    #GNB
    filename2="GNB_model.sav"
    loaded_model2 = pickle.load(open(filename2, 'rb'))
    result2 = loaded_model2.predict([df])
    result3 = loaded_model2.predict_proba([df])
    print("GNB",result)  
    rs=['Not Placed',"Placed"]
    prd=np.argmax(result3)
    print("GNB",prd)


    upres=Dataset_tbl.objects.get(id=idno)
    upres.Result=rs[result[0]]+percnt+"%"
    upres.save()
    obj=Admin_tbl.objects.all()
    data=Dataset_tbl.objects.all()
    return render(request,'home_t.html',{'data':obj,'dataset':data})


#*******************************************8888888888/******************************************************
def Works(request):
    wobj = Works_tbl.objects.all()
    return render(request, 'student/works.html', {'data': wobj})

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
        obj=tutorReg_tbl.objects.filter(username=username,passwd=password)
        if obj:
            request.session['username']=username
            request.session['password']=password
            return redirect("/tutor/home")
            # return render(request,'tutor/tutor_homepage.html')
        else:
            request.session['username']=""
            request.session['password']=""

            return render(request,'tutor/signin.html',{'lmsg':"check your data"})
    else:
        msg=" "
        return render(request,'tutor/signin.html',{'lmsg':msg})
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
def gmeet(request):
    return render(request,"gmeet.html")
def seeassignt(request):
    if request.method=="POST": 
        crs=request.POST.get("course1")
        dat=stdworks_tbl.objects.all()
        crs1=course.objects.all()
        return render(request,'tutor/seeassignt.html',{'data':dat,'crs':crs1})
    else:
        crs=course.objects.all()
        return render(request, 'tutor/seeassignt.html',{"crs":crs}) 

       
