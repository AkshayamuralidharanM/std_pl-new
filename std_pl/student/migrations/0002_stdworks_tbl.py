# Generated by Django 4.0 on 2022-05-13 10:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('student', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='stdworks_tbl',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('assname', models.CharField(max_length=100)),
                ('assign', models.FileField(upload_to='assign')),
                ('assdate', models.DateField()),
                ('stdcourse', models.CharField(max_length=100)),
                ('stdname', models.CharField(max_length=100)),
            ],
        ),
    ]
