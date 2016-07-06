__author__ = 'geco'

from os import listdir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

images_path = "./graphs/"
dataset_path = "./dataset/"
file_list = listdir(dataset_path)
sizes = {
    "8": [],
    "y": [],
    "t:": [],
    "x": [],
    "Hal": [],
    "su": [],
}
colors = {
    "8": 'g',
    "y": 'r',
    "t:": 'b',
    "x": 'y',
    "Hal": 'w',
    "su": 'p',
}
min_size = 9999
max_size = 0
for filename in file_list:
    with open(dataset_path+"/"+filename,'r') as f:
        period = f.readline()
        xdim, ydim = f.readline().split()
        _ = f.readline()
        howmanypoints = sum([1 for line in f])
        # print filename,": ",howmanypoints
        class_name = filename.split("-")[0][3:]
        if class_name in sizes:
            min_size = min(min_size, howmanypoints)
            max_size = max(max_size, howmanypoints)
            sizes[class_name].append(howmanypoints)

dif = max_size-min_size
num_buckets = 100

bins = np.linspace(min_size, max_size)

for cur_cla_name,cur_cla_sizes in sizes.iteritems():
    n, bins, patches = plt.hist(cur_cla_sizes, bins=bins,  alpha=0.75, label = cur_cla_name)
plt.xlabel('Longitud de muestra')
plt.ylabel('Cantidad de muestras')
plt.title('Cantidad de muestras por clase y longitud sin bucketing')
plt.grid(True)
plt.legend()
plt.savefig("%s/histograma_%d"%(images_path,num_buckets))
plt.close()

num_buckets = 40
bins = np.linspace(min_size, max_size, num_buckets)
for cur_cla_name,cur_cla_sizes in sizes.iteritems():
    n, bins, patches = plt.hist(cur_cla_sizes, bins=bins,  alpha=0.75, label = cur_cla_name)
plt.xlabel('Longitud de muestra')
plt.ylabel('Cantidad de muestras')
plt.title('Cantidad de muestras por clase y longitud con %d buckets'%(num_buckets))
plt.grid(True)
plt.legend()
plt.savefig("%s/histograma_%d"%(images_path,num_buckets))
plt.close()

num_buckets = 15
bins = np.linspace(min_size, max_size, num_buckets)
for cur_cla_name,cur_cla_sizes in sizes.iteritems():
    n, bins, patches = plt.hist(cur_cla_sizes, bins=bins,  alpha=0.75, label = cur_cla_name)
plt.xlabel('Longitud de muestra')
plt.ylabel('Cantidad de muestras')
plt.title('Cantidad de muestras por clase y longitud con %d buckets'%(num_buckets))
plt.grid(True)
plt.legend()
plt.savefig("%s/histograma_%d"%(images_path,num_buckets))
plt.close()


num_buckets = 5
bins = np.linspace(min_size, max_size, num_buckets)
for cur_cla_name,cur_cla_sizes in sizes.iteritems():
    n, bins, patches = plt.hist(cur_cla_sizes, bins=bins,  alpha=0.75, label = cur_cla_name)
plt.xlabel('Longitud de muestra')
plt.ylabel('Cantidad de muestras')
plt.title('Cantidad de muestras por clase y longitud con %d buckets'%(num_buckets))
plt.grid(True)
plt.legend()
plt.savefig("%s/histograma_%d"%(images_path,num_buckets))
plt.close()