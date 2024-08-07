from django.shortcuts import render,redirect
from numpy.testing._private.utils import integer_repr
from sklearn.feature_extraction.text import TfidfVectorizer
from . models import Admin_tbl,Dataset_tbl
from django.http import HttpResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from student.models import User_mp
from hod.models import hodReg_tbl
from tutor.models import tutorReg_tbl
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np 
import pickle
from django.conf import settings
from django.core.mail import send_mail
# Create your views here.

def index(request):
    return render(request,'index.html')
# #Admin Home Page
# def home(request):
#     return render(request,'admin_homepage.html')
#login page
def login(request):
    if request.method=="POST":
        username=request.POST.get("user")
        password=request.POST.get("passw")
        obj=Admin_tbl.objects.filter(username=username,password=password)
        if obj:
            request.session['username']=username
            request.session['password']=password
            return redirect("/adminApp/home")
            # return render(request,'tutor/tutor_homepage.html')
        else:
            request.session['username']=""
            request.session['password']=""

            return render(request,'adminApp/admin_login.html',{'lmsg':"check your data"})
    else:
        msg=" "
        return render(request,'adminApp/admin_login.html',{'lmsg':msg})
   


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
            return render(request,'home.html',{'data':obj,'dataset':data})
            
        else:
            msg="Not Successfully"
            return render(request,'Dataset_reg.html',{'msg':msg})    
    msg=" "        
    return render(request,'Dataset_reg.html',{'msg':msg})
def home(request):
    usr=request.session["username"]
    passw=request.session["password"]
    if usr=="" and passw=="":
        return  redirect("/logout")    
    else:
        obj=Admin_tbl.objects.filter(username=usr,password=passw)
        data=Dataset_tbl.objects.all()
        return render(request,'adminApp/admin_homepage.html',{'data':obj,'dataset':data})
def logout(request):
    request.session['username']=""
    request.session['password']=""
    return render(request,'adminApp/admin_login.html')  
def update(request):
    if request.method=="POST":
        usr=request.POST.get("user") 
        passw=request.POST.get("passw")
        #idno=request.POST.get('idno')
        s=request.session['username']
        p=request.session['password']
        obj3=Admin_tbl.objects.filter(username=s,password=p)
        for l in obj3:
            idn=l.id
        obj=Admin_tbl.objects.get(id=idn)
        obj.username=usr
        obj.password=passw
        obj.save()
        obj2=Admin_tbl.objects.filter(username=usr,password=passw)
        msg="updattion is successfully saved"
        return render(request,'home.html',{'data':obj2,'msg':obj})
    else:
        return redirect('/home')    
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
    return render(request,'home.html',{'data':obj,'dataset':data})
            




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
def delt(request):
    idno=request.GET.get("idn")       
    obj=Dataset_tbl.objects.get(id=idno)
    obj.delete()
    return redirect('/home')

# Student Data Confirm
def mp_confirm(request):
    print(request.GET.get('idn'))
    if request.GET.get("idn"):
        idno = request.GET.get("idn")
        mpobj = User_mp.objects.get(id=idno)
        mpobj.vrf="confirm"
    

        mpobj.save()
        print("update")
        mpobj = User_mp.objects.get(id=idno)
        passw=mpobj.passwd
        usern=mpobj.username
    #sending mail
        subject = "Username and Password"
        msg ="Your Username:"+usern+"\n Password:"+passw+"\n Login using this link http://127.0.0.1:8000/student/"
        to = usern
        res = send_mail(subject,msg,settings.EMAIL_HOST_USER,[to])
        if res:
            l = "mail send successfully"
        else:
            l ="mail not send"
            return redirect("/adminApp/home")
    return redirect("/adminApp/home")        
def delt(request):
    idno=request.GET.get("idn")
    print(idno)
    mpobj=User_mp.objects.get(id=idno)
    if mpobj:
        mpobj.delete()
    return redirect("/adminApp/home")
    




def studdelt(request):
    idno=request.GET.get("idno")
    
    mpobj=Dataset_tbl.objects.get(id=idno)
    if mpobj:
        mpobj.delete()

    obj=Admin_tbl.objects.all()
    data=Dataset_tbl.objects.all()
    return render(request,'home.html',{'data':obj,'dataset':data})
                

# Admin Home Page
def home(request):
    return render(request,"adminApp/admin_homepage.html")

def hoddelt(request):
    idno=request.GET.get("idn")
    print(idno)
    mpobj=hodReg_tbl.objects.get(id=idno)
    if mpobj:
        mpobj.delete()
    return redirect("/adminApp/home")

def tutdelt(request):
    idno=request.GET.get("idn")
    print(idno)
    mpobj=tutorReg_tbl.objects.get(id=idno)
    if mpobj:
        mpobj.delete()
    return redirect("/adminApp/home")

            







    




