import csv
with open('results.csv', 'r') as f:
    f.readline()
    result_info = {}
    for row in csv.DictReader(f, delimiter=' '):
        print row
        result_info[row['name']] = row
