#! /usr/bin/env python
import requests

class WebService(object):
    def __init__(self):
        pass

    # def set_sorted_filenames(self, stack, sorted_filenames):
    #     self.convert_to_request('set_sorted_filenames', **kwargs)
    #
    # def align(self, stack, first_section, last_section, bad_sections):
    #     self.convert_to_request('align', **kwargs)

    def convert_to_request(self, name, **kwargs):
        r = requests.get('http://gcn-20-33.sdsc.edu:5000/'+name, params=kwargs)
        print r.url
        return r.json()
