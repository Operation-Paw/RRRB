from django.db import models

# Create your models here.
#Each model is a Python class that subclasses django.db.models.Model.
class Employee(models.Model):
 empcode=models.IntegerField( )
 name=models.CharField(max_length=30)
 mobile=models.CharField(max_length=10)
 email=models.EmailField(max_length=50)