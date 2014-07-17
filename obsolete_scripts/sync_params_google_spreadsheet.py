import gspread
import getpass
import os, sys, csv
import argparse

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description="Sync parameters csv file from the Google spreadsheet")

parser.add_argument("-u", "--username", type=str, help="google username (default: %(default)s)", default='cyc3700@gmail.com')
parser.add_argument("-p", "--params_file", type=str, help="parameters csv file (default: %(default)s)", default='/oasis/projects/nsf/csd181/yuncong/Brain/params.csv')
args = parser.parse_args()
        
username = "cyc3700@gmail.com"
password = getpass.getpass()

params_file = 'params.csv'

docid = "1S189da_CxzC3GKISG3hZDG0n7mMycC0v4zTiRJraEUE"
client = gspread.login(username, password)
spreadsheet = client.open_by_key(docid)
for i, worksheet in enumerate(spreadsheet.worksheets()):
    with open(params_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(worksheet.get_all_values())
