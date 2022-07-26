import csv
import numpy as np

name = 'report_256_8.csv'


wfile = open('short_' + name, 'w', newline='')
file = open(name, 'r', newline='')
writer = csv.writer(wfile, delimiter=',')

reader = csv.reader(file, delimiter=',')
next(reader)

count = 0

dict_keys = ['iframe PSNR', 'iframe bits', 'PSNR', 'total bits', 'my', 'mz', 'ry', 'rz']
dict_val = [[], [], [], [], [], [], [], [], ]
log_dict = dict(zip(dict_keys, dict_val))

writer.writerow(['seq name', 'iframe PSNR', 'iframe bits', 'PSNR', 'total bits', 'my', 'mz', 'ry', 'rz'])

for row in reader:
    if count % 10 == 0:
        log_dict['iframe PSNR'].append(float(row[1]))
        log_dict['iframe bits'].append(float(row[2]))

    else:
        log_dict['PSNR'].append(float(row[1]))
        log_dict['total bits'].append(float(row[2]))
        log_dict['my'].append(float(row[3]))
        log_dict['mz'].append(float(row[4]))
        log_dict['ry'].append(float(row[5]))
        log_dict['rz'].append(float(row[6]))

    count += 1

    if count % 60 == 0:
        summary = [row[0].split('/')[0]]

        for key in dict_keys:
            # print(key)
            summary.append(np.mean(log_dict[key]))
            # print(log_dict[key])

        print(summary)
        writer.writerow(summary)

        log_dict = dict(zip(dict_keys, dict_val))
