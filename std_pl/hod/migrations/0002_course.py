# Generated by Django 4.0.2 on 2022-05-13 00:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hod', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='course',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cname', models.CharField(max_length=50)),
                ('duration', models.CharField(max_length=50)),
            ],
        ),
    ]
