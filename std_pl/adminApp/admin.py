from django.contrib import admin
from adminApp.models import Admin_tbl
from student.models import User_mp
# Register your models here.

admin.site.register(Admin_tbl)
admin.site.register(User_mp)