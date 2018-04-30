from django.shortcuts import render
from django.views.generic import TemplateView
from demo.models import Mac_device

path = "device_list.txt"
# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)


def homepage(request):
	mac_list = []
	context = {}
	mac_list = get_device_list(path)
	context['mac_devices'] = mac_list
	return render(request, 'index.html',context)


def get_device_list(path):
	device_list = []

	with open(path) as f:
		for line in f:
			cols = line.rstrip().split('\t')
			device_list.append(Mac_device(mac_addr=cols[0], device_type=cols[1]))

	return device_list

def refresh(request):
	mac_list = []
	context = {}
	mac_list = get_device_list(path)
	context['mac_devices'] = mac_list
	return render(request, 'mac_devices.json', context, content_type='application/json')
