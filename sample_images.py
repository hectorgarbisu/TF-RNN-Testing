__author__ = 'geco'
from os import listdir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
images_path = "./sample_images/"
dataset_path = "./dataset/"
file_list = listdir(dataset_path)


for filename in file_list:
    f = open(dataset_path+"/"+filename,'r')
    period = f.readline()
    xdim, ydim = f.readline().split()
    _ = f.readline()
    data = list()
    xs = []
    ys = []
    for line in f:
        xi,yi = line.split()
        xs.append(float(xi)/300)
        ys.append(1-1*float(yi)/300)
        "coordinates between -1 and 1"
    f.close()
    _, ((b,c),(e,d)) = plt.subplots(2,2)
    b.plot(xs,ys,"-")
    b.set_title("Firma")
    c.plot(xs[::1],ys[::1],".")
    c.set_title("Puntos")
    # plt.savefig("%s/%s"%(images_path,))
    current_size = len(xs)
    jj = 0
    """ Return test set and its labels """
    while(current_size<100):
        newx = (xs[jj]+xs[jj+1])/2
        newy = (ys[jj]+ys[jj+1])/2
        xs.insert(jj+1,newx)
        ys.insert(jj+1,newy)
        current_size += 1
        " loop through the whole sample as many times as needed "
        jj = (jj+2)%(current_size-1)
    d.plot(xs[::1],ys[::1],".")
    d.set_title("Puntos interpolados")
    # plt.savefig("%s/%s"%(images_path,))

    xs = [xx-xs[0] for xx in xs]
    ys = [yy-ys[0] for yy in ys]
    # print centered_irregular_signatures

    a0 = 0.0000000001 #
    xmax = max(abs(max(xs)),abs(min(xs)))+a0
    ymax = max(abs(max(ys)),abs(min(ys)))+a0
    "Fitting is made so the first point keeps at 0,0"
    xsf = 1/xmax # x_scalation_factor
    ysf = 1/ymax # y_scalation_factor
    xsf = ysf = min(xsf,ysf)
    # assert(xsf>0.0001)
    # assert(ysf>0.0001)
    xs = [xx*xsf for xx in xs]
    ys = [yy*ysf for yy in ys]
    e.plot(xs[::1],ys[::1],".")
    e.set_title("Centrado y escalado")
    e.axis([-1,1,-1,1])
    plt.savefig("%s/%s"%(images_path,filename.split(".")[0]))
    plt.close()