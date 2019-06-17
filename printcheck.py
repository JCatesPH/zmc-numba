#%%
import numpy as np
import datetime

beg = 1
end = 51
spacing = 50

print('om')
for i in np.linspace(beg, end, spacing):
    print('%5.3f' % (1.98 + i * 0.05 / spacing))

arr = np.arange(0,50)


#%%
filename = str(datetime.datetime.today()) + '.csv'

with open(filename, 'w') as csvout:
    csvout.write('om')

    j = 0
    for i in np.linspace(beg, end, spacing):
        csvout.write('\n')
        csvout.write(str(i))
        csvout.write(',')
        csvout.write(str(arr[j]))
        j = j + 1

#%%
