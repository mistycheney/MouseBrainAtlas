#! /usr/bin/env python
import requests
from metadata import *

def web_services_request(self, name, **kwargs):
    r = requests.get('http://gcn-20-33.sdsc.edu:5000/'+name, params=kwargs)
    return r.json()

class WebService(object):
    def __init__(self, server_ip, port=5000):
        self.server_str = 'http://' + server_ip + ':' + str(port)

    def convert_to_request(self, name, **kwargs):
        r = requests.get(self.server_str + '/'+name, params=kwargs)
        return r.json()
