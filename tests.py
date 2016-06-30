__author__ = 'geco'
import RNN
import time
import numpy as np
from dataset_loader import dataset_loader
from tensorflow.python.framework import ops


dataset_path = "./dataset/"
result_path = "./results/"

##################################
######## FIXED PARAMETERS ########
##################################

num_epochs = 200
alpha = 0.01
batch_size = 100
nW_hidden = 60
test_repeat = 10
e_s = 10 # error samples

num_clases = 0
sample_size_idx = 0
fill_method_idx = 0
dl = dataset_loader(dataset_path)
dl.load(fixed_sig_size=100) #mock load
h_u_l = dl.get_labels_to_hot_dict() #homogenized unique labels so that every training use the same

##################################
########### VARIABLES ############
##################################

"sample sizes' = [int(100*(1.3**ii)) for ii in range(10)]"
"fill_methods = [interpolation, left_padding, right_padding]"

fill_methods = ["interpolation", "left_padding", "right_padding"]
# fill_methods = ["left_padding"]
sample_sizes = [100, 130, 169, 219, 285, 371, 482, 627, 815, 1060]
decreasing = False
if decreasing:
    sample_sizes = sample_sizes.reverse()


errors = np.zeros([len(sample_sizes),len(fill_methods),e_s])
times = np.zeros([len(sample_sizes),len(fill_methods),e_s])
confusion_matrices = np.zeros([len(sample_sizes),len(fill_methods),len(h_u_l),len(h_u_l)])

def single_test(sample_size,fill_method):
    input_size = 2
    dl = dataset_loader(dataset_path)
    dl.load(fixed_sig_size=sample_size, fill_method=fill_method) #mock load
    la_to_ho = h_u_l
    # la_to_ho = dl.get_labels_to_hot_dict()
    num_classes = len(la_to_ho)
    rnn = RNN.RNN(input_size,nW_hidden,sample_size,num_classes)
    # Get batch and its labels
    cctime = time.time()
    train(rnn,dl)
    print test(rnn,dl,la_to_ho)," ,",time.time()-cctime," s."
    rnn.sess.close()
    ops.reset_default_graph()

##################
##### TRAIN ######
####choo#choo#####

def train(rnn,dl):
    err = 0
    jj=0
    prev_time = time.time()
    for ii in range(num_epochs):
        batch,_,hotone_labels = dl.next_2d_batch(batch_size)
        rnn.feed_batch(batch,hotone_labels)
        train_time = time.time() - prev_time
        err = rnn.error(batch,hotone_labels)
        if (ii % (num_epochs//e_s)) == 0:
            # print ii,"/",num_epochs
            errors[sample_size_idx][fill_method_idx][jj] += err/test_repeat
            times[sample_size_idx][fill_method_idx][jj] += train_time/test_repeat
            jj+=1

##################
###### TEST ######
##################

def test(rnn,dl,la_to_ho):
    total = 0
    error = 0
    for signature, expected_label in dl.get_test_set():
        flat_signature = np.array(signature).flatten()
        predicted_label = rnn.categorize(flat_signature)
        max0 = np.argmax(la_to_ho[expected_label])
        max1 = np.argmax(predicted_label)
        total += 1
        # Build confusion matrix
        add_confusion_matrix(max0,max1)
        if(max0!=max1):
            error += 1
    return "errores: %d / %d : %f%%acierto"%(error,total,100*(1-float(error)/total))


def add_confusion_matrix(max0, max1, la_to_ho=h_u_l):
    confusion_matrices[sample_size_idx][fill_method_idx][max0][max1]+=1


def print_results_to_file():
    error_file = open("%s%d-%s-error"%(result_path,sample_size,fill_method),"w")
    for ii in range(e_s):
        time_and_error = (times[sample_size_idx][fill_method_idx][ii],
                          errors[sample_size_idx][fill_method_idx][ii])
        error_file.write("%f:%f\n"%time_and_error)
        print time_and_error
    error_file.close()
    confusion_file = open("%s%d-%s-confumat"%(result_path,sample_size,fill_method),"w")
    confusion_file.write(" ".join(h_u_l.keys())+"\n")
    for ii in range(6):
        for jj in range(6):
            confusion_file.write(" "+str(confusion_matrices[sample_size_idx][fill_method_idx][ii][jj]))
        confusion_file.write("\n")
    confusion_file.close()


for sample_size_idx,sample_size in enumerate(sample_sizes):
    for fill_method_idx,fill_method in enumerate(fill_methods):
        case_start_time = time.time()
        print("Case: sample_size=",sample_size," fill_method=",fill_method)
        for ii in range(test_repeat):
            single_test(sample_size,fill_method)
        print(time.time()-case_start_time,"s. for case: sample_size=",sample_size," fill_method=",fill_method)
        print times[sample_size_idx][fill_method_idx]
        print h_u_l.keys()
        print confusion_matrices[sample_size_idx][fill_method_idx]
        print_results_to_file()
