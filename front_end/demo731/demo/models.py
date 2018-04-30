from django.db import models

# Create your models here.

class Mac_device(models.Model):
	mac_addr = models.CharField( max_length = 50,default= ' ')
	device_type = models.CharField( max_length = 50,default= ' ')