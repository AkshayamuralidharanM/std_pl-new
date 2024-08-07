from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class User_mp(models.Model):
    mpname=models.CharField(max_length=100)
    gender = models.CharField(max_length=100)
    dob=models.EmailField(max_length=100)
    address=models.CharField(max_length=100)
    phoneno=models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    username=models.CharField(max_length=100)
    passwd=models.CharField(max_length=100)
    gn=models.CharField(max_length=100)
    gpn=models.CharField(max_length=100)
    vrf = models.CharField(max_length=100)
    mcs=models.CharField(max_length=100)
    sname=models.CharField(max_length=100)
    batch=models.CharField(max_length=100)

    def __str__(self):
        return self.mpname


# Course Section
class Course(models.Model):
    user = models.ForeignKey(User_mp,on_delete=models.CASCADE)
    cname = models.CharField(max_length=30)
    cfee = models.IntegerField()
    cdetails = models.TextField()
# Student Assignment Section
class stdworks_tbl(models.Model):
    assname = models.CharField(max_length=100)
    assign  = models.FileField(upload_to="assign")
    assdate = models.DateField(auto_now=True)
    stdcourse = models.CharField(max_length=100)
    stdname = models.CharField(max_length=100)

    def __str__(self):
        return self.assname


 