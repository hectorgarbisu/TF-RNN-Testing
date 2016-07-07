__author__ = 'geco'
from os import listdir

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
results_path = "./optimized_results/"
graphs_path = "./aditional_graphs/"
file_list = listdir(results_path)

def savefig(str):
    print "saving %s"%str
    plt.savefig(str)
    # plt.show()

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
    savefig("%s/%s-%s-error"%(graphs_path,size,method))
    plt.close()
    # print x,y



def confu_graph(size,method,file):
    # COUNT ERRORS
    labels = file.readline()
    confumat = []
    for line in file:
        confumat.append([float(a) for a in line.split()])
    confumat = np.array(confumat)
    for ii in range(len(confumat)):
        confumat[ii][ii] = 0.0
    confuflat = confumat.flatten().tolist()
    # print confuflat
    bins = range(20)
    plt.hist(confuflat,bins=bins,  range=[1,20],alpha=0.75)
    plt.xlim(1,15)
    savefig("%s/%s-%s-confugraph"%(graphs_path,size,method))
    plt.close()

def confu_multigraph(confufiles):
    # COUNT ERRORS
    for _,method,filename in confufiles:
        file = open(results_path+"/"+filename,"r")
        labels = file.readline()
        confumat = []
        for line in file:
            confumat.append([float(a) for a in line.split()])
        confumat = np.array(confumat)
        for ii in range(len(confumat)):
            confumat[ii][ii] = 0.0
        confuflat = confumat.flatten().tolist()
        # print confuflat
        bins = range(20)
        plt.hist(confuflat,bins=bins,  range=[1,20],alpha=0.75, label = method)
        plt.xlim(1,15)
    plt.legend()
    savefig("%s/multi-confugraph"%(graphs_path))
    plt.close()
    pass

def roc_space(confufiles):
 # COUNT ERRORS
    TPRss = []
    FPRss = []
    methods = []
    for _,method,filename in confufiles:
        methods.append(method)
        file = open(results_path+"/"+filename,"r")
        label_names = file.readline().split()
        rocs = dict()
        confumat = []
        for line in file:
            confumat.append([float(a) for a in line.split()])
        row_totals = [sum(row) for row in confumat]
        total = sum(row_totals)
        confumat = np.array(confumat)
        TPRs = []
        FPRs = []
        for ii in range(len(confumat)):
            TPRs.append(confumat[ii][ii]/row_totals[ii])
            col_total = sum([confumat[jj][ii] for jj in (range(len(confumat)))])
            FPRs.append((col_total-confumat[ii][ii])/(total-row_totals[ii]))
        confuflat = confumat.flatten().tolist()
        TPRss.append(TPRs)
        FPRss.append(FPRs)
    plt.plot(FPRss[0],TPRss[0],"o",label=methods[0])
    plt.plot(FPRss[1],TPRss[1],"o",label=methods[1])
    plt.legend()
    plt.title("Espacio ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    # plt.plot([0,1],[0,1],"r-")
    # plt.axis([0,0.1,0,1])
    savefig("%s/scatter_roc"%(graphs_path))

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
        # confu_graph(size,method,file)
# error_multigraph(error_files)
roc_space(confu_files)
# confu_multigraph(confu_files)


