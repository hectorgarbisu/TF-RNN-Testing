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
    #GET CE ERRORS
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
    #PRINT INTERPOLATION ERRORS
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
    #PRINT RPADDING ERRORS
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
    #PRINT LPADDING ERRORS
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

def confu_multigraph(size_method_filenames):
    interpolation_confumats = dict()
    rpadding_confumats = dict()
    lpadding_confumats = dict()
    classes = []
    for size,method,filename in size_method_filenames:
        file = open(results_path+"/"+filename)
        classes = file.readline().strip().split()
        if method=="interpolation":
            interpolation_confumats[size] = [[float(a) for a in line.strip().split(" ")] for line in file]
        elif method=="left_padding":
            lpadding_confumats[size] = [[float(a) for a in line.strip().split(" ")] for line in file]
        elif method=="right_padding":
            rpadding_confumats[size] = [[float(a) for a in line.strip().split(" ")] for line in file]

    #CLASSIFICATION ERROR
    class_errors = []
    for size in interpolation_confumats.iterkeys():
        ic = interpolation_confumats[size]
        lc = lpadding_confumats[size]
        rc = rpadding_confumats[size]
        diagonalic = sum([ic[ii][ii] for ii in range(len(ic))])
        diagonallc = sum([lc[ii][ii] for ii in range(len(lc))])
        diagonalrc = sum([rc[ii][ii] for ii in range(len(rc))])
        totalic = sum([sum(icline) for icline in ic])
        totallc = sum([sum(lcline) for lcline in lc])
        totalrc = sum([sum(rcline) for rcline in rc])
        class_errors.append((float(size), diagonalic/totalic, diagonallc/totallc, diagonalrc/totalrc))
    class_errors = sorted(class_errors,key=lambda siz: siz[0])

    # show average classification errors by method and size
    ax1 = plt.figure().add_subplot(111)
    ax1.set_ylabel("Error de clasificacion")
    ax1.set_xlabel("Longitud de muestra")
    plt.plot([ce[0] for ce in class_errors],[ce[1] for ce in class_errors],".-",label="interpolacion")
    plt.plot([ce[0] for ce in class_errors],[ce[2] for ce in class_errors],".-",label="relleno por la izquierda")
    plt.plot([ce[0] for ce in class_errors],[ce[3] for ce in class_errors],".-",label="relleno por la derecha")
    plt.legend(loc=5)
    plt.savefig("%s/classification-errors-by-size"%graphs_path)

    # print classification errors by method, size, and class # [(class, size, error)]
    interpolation_errors_by_class = []
    lpadding_errors_by_class = []
    rpadding_errors_by_class = []
    for size,confumat in interpolation_confumats.iteritems():
        for idx,row in enumerate(confumat):
            size_class_errors = (idx,float(size), row[idx]/sum(row))
            interpolation_errors_by_class.append(size_class_errors)
    interpolation_errors_by_class = sorted(interpolation_errors_by_class, key=lambda siz: siz[1])
    interpolation_errors_by_class = sorted(interpolation_errors_by_class, key=lambda siz: siz[0])
    ax1 = plt.figure().add_subplot(111)
    ax1.set_ylabel("Error de clasificacion")
    ax1.set_xlabel("Longitud de muestra")
    for class_idx,class_name in enumerate(classes):
        current_class_errors = interpolation_errors_by_class[10*class_idx:10*class_idx+9]
        plt.plot([err[1] for err in current_class_errors],[err[2] for err in current_class_errors],".-",label=class_name)
    plt.legend(loc=3)
    plt.title("Error de clasificacion con interpolacion por clase")
    plt.savefig("%s/classification-errors-with-interpolation-by-class-and-size"%graphs_path)

    for size,confumat in lpadding_confumats.iteritems():
        for idx,row in enumerate(confumat):
            size_class_errors = (idx,float(size), row[idx]/sum(row))
            lpadding_errors_by_class.append(size_class_errors)
    lpadding_errors_by_class = sorted(lpadding_errors_by_class, key=lambda siz: siz[1])
    lpadding_errors_by_class = sorted(lpadding_errors_by_class, key=lambda siz: siz[0])
    ax1 = plt.figure().add_subplot(111)
    ax1.set_ylabel("Error de clasificacion")
    ax1.set_xlabel("Longitud de muestra")
    for class_idx,class_name in enumerate(classes):
        current_class_errors = lpadding_errors_by_class[10*class_idx:10*class_idx+9]
        plt.plot([err[1] for err in current_class_errors],[err[2] for err in current_class_errors],".-",label=class_name)
    plt.legend(loc=4)
    plt.title("Error de clasificacion con relleno por la izquierda por clase")
    plt.savefig("%s/classification-errors-with-leftpadding-by-class-and-size"%graphs_path)

    for size,confumat in rpadding_confumats.iteritems():
        for idx,row in enumerate(confumat):
            size_class_errors = (idx,float(size), row[idx]/sum(row))
            rpadding_errors_by_class.append(size_class_errors)
    rpadding_errors_by_class = sorted(rpadding_errors_by_class, key=lambda siz: siz[1])
    rpadding_errors_by_class = sorted(rpadding_errors_by_class, key=lambda siz: siz[0])
    ax1 = plt.figure().add_subplot(111)
    ax1.set_ylabel("Error de clasificacion")
    ax1.set_xlabel("Longitud de muestra")
    for class_idx,class_name in enumerate(classes):
        current_class_errors = rpadding_errors_by_class[10*class_idx:10*class_idx+9]
        plt.plot([err[1] for err in current_class_errors],[err[2] for err in current_class_errors],".-",label=class_name)
    plt.legend(loc=4)
    plt.title("Error de clasificacion con relleno por la derecha por clase")
    plt.savefig("%s/classification-errors-with-rightpadding-by-class-and-size"%graphs_path)

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
confu_multigraph(confu_files)


# plt.savefig('myfig')