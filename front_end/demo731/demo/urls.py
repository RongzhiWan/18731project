from django.conf.urls import url
from demo import views

urlpatterns = [
    url(r'^$', views.homepage,name = 'homepage'),
    url(r'^refresh$', views.refresh, name = "refresh"),
]