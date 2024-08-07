from django.db import models

# Create your models here.
class Admin_tbl(models.Model):
    username=models.CharField(max_length=100)
    password=models.CharField(max_length=100)

class Dataset_tbl(models.Model):
    sname=models.CharField(max_length=100)
    g=models.CharField(max_length=100)
    ssc_p=models.DecimalField(max_digits=9,decimal_places=2)
    ssc_b=models.CharField(max_length=100)
    hsc_p=models.DecimalField(max_digits=9,decimal_places=2)
    hsc_b=models.CharField(max_length=100)
    hsc_s=models.CharField(max_length=100)
    degree_p=models.DecimalField(max_digits=9,decimal_places=2)
    degree_t=models.CharField(max_length=100)
    workex=models.CharField(max_length=100)
    etest_p=models.DecimalField(max_digits=9,decimal_places=2)
    specialisation=models.CharField(max_length=100)
    mba_p=models.DecimalField(max_digits=9,decimal_places=2)
    Department=models.CharField(max_length=100)
    Result=models.CharField(max_length=100)
    salary=models.DecimalField(max_digits=9,decimal_places=2)









    

