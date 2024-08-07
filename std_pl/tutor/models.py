from django.db import models

# Create your models here.
# Create your models here.
class tutorReg_tbl(models.Model):
    tname=models.CharField(max_length=100)
    gender = models.CharField(max_length=100)
    address=models.CharField(max_length=100)
    phoneno=models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    username=models.CharField(max_length=100)
    passwd=models.CharField(max_length=100)
    dpt=models.CharField(max_length=100)
    desi=models.CharField(max_length=100)
    photo=models.FileField(upload_to="tutorpic")
    def __str__(self):
        return self.tname
class Notes_tbl(models.Model):
    title=models.CharField(max_length=100)
    # dt = models.CharField(max_length=100)
    dt=models.DateField(auto_now=True)
    url = models.FileField(upload_to="notes")
    crs = models.CharField(max_length=100)
    std = models.CharField(max_length=100)
    status = models.CharField(max_length=100)
    def __str__(self):
        return self.title
class Works_tbl(models.Model):
    notes=models.FileField(upload_to="works")
    wt=models.CharField(max_length=100)
    wdt=models.DateField(auto_now=True)
    course=models.CharField(max_length=100)
    std=models.CharField(max_length=100)
    def __str__(self):
        return self.wt

class course(models.Model):
    cname=models.CharField(max_length=50)
    duration=models.CharField(max_length=50)
    def __str__(self):
        return self.cname    
