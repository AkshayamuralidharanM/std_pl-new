from django.contrib import admin
from tutor.models import  tutorReg_tbl,course,Notes_tbl,Works_tbl
# Register your models here.
admin.site.register(tutorReg_tbl)
admin.site.register(course)
admin.site.register(Notes_tbl)
admin.site.register(Works_tbl)