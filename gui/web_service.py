#! /usr/bin/env python
import requests

class WebService(object):
    def __init__(self, stack):
        self.stack = stack

    def set_sorted_filenames(self, sorted_filenames):
        self.convert_to_request('set_sorted_filenames', **kwargs)

    def align(self, first_section, last_section, bad_sections):
        self.convert_to_request('align', **kwargs)

    def convert_to_request(self, name, kwargs):
        r = requests.get('http://gcn-20-32.sdsc.edu:5000/'+name, params=kwargs)
        print r.url
        return r.json()
