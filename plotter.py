__author__ = 'geco'
from os import listdir

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
results_path = "./results/"
graphs_path = "./graphs/"
file_list = listdir(results_path)

def error_graph(size,method,file):
    x = []
    y = []
    for line in file:
        a,b = line.split(":")
        x.append(float(a))
        y.append(float(b))

    ax1 = plt.figure().add_subplot(111)
    ax1.set_title("%s-%s"%(size,method))
    ax1.set_ylabel("error (cross entropy)")
    ax1.set_xlabel("tiempo (s)")
    plt.plot(x,y,".-",label="error")

    # plt.show()
    plt.savefig("%s/%s-%s-error"%(graphs_path,size,method))
    plt.close()
    # print x,y

def error_multigraph(size_method_filenames):
    interpolation_errors = dict()
    lpadding_errors = dict()
    rpadding_errors = dict()
    for size,method,filename in size_method_filenames:
        file = open(results_path+"/"+filename)
        if method=="interpolation":
            interpolation_errors[size] = [[a for a in line.split(":")] for line in file]
        elif method=="left_padding":
            lpadding_errors[size] = [[a for a in line.split(":")] for line in file]
        elif method=="right_padding":
            rpadding_errors[size] = [[a for a in line.split(":")] for line in file]
    ax1 = plt.figure().add_subplot(111)
    ax1.set_ylabel("error (cross entropy)")
    ax1.set_xlabel("tiempo (s)")
    for key,val in interpolation_errors.iteritems():
        x = [float(a[0]) for a in val]
        y = [float(a[1]) for a in val]
        plt.plot(x,y,".-",label=key)
        plt.legend()
        plt.title("Error con interpolacion")
    # plt.show()
    plt.savefig("%s/interplation-error"%(graphs_path))
    ax1 = plt.figure().add_subplot(111)
    ax1.set_ylabel("error (cross entropy)")
    ax1.set_xlabel("tiempo (s)")
    for key,val in rpadding_errors.iteritems():
        x = [float(a[0]) for a in val]
        y = [float(a[1]) for a in val]
        plt.plot(x,y,".-",label=key)
        plt.legend()
        plt.title("Error con relleno por la derecha")
    # plt.show()
    plt.savefig("%s/right_padding-error"%(graphs_path))
    ax1 = plt.figure().add_subplot(111)
    ax1.set_ylabel("error (cross entropy)")
    ax1.set_xlabel("tiempo (s)")
    for key,val in lpadding_errors.iteritems():
        x = [float(a[0]) for a in val]
        y = [float(a[1]) for a in val]
        plt.plot(x,y,".-",label=key)
        plt.legend()
        plt.title("Error con relleno por la izquierda")
    # plt.show()
    plt.savefig("%s/left_padding-error"%(graphs_path))




def confu_graph(size,method,file):
    pass

def confu_multigraph(files):
    pass

error_files = []
confu_files = []
for filename in file_list:
    size,method,res_type = filename.split("-")
    file = open(results_path+"/"+filename,"r")
    if(res_type=="error"):
        error_files.append((size,method,filename))
        # error_graph(size,method,file) #indvidual graph for this
    elif(res_type=="confumat"):
        confu_files.append((size,method,filename))
        confu_graph(size,method,file)
error_multigraph(error_files)
confu_multigraph(confu_files)


# plt.savefig('myfig')