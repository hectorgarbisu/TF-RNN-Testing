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

num_epochs = 2000
alpha = 0.01
batch_size = 20
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
sample_size = 100


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
            print ii,"/",num_epochs, "err: ", err
            # errors[sample_size_idx][fill_method_idx][jj] += err/test_repeat
            # times[sample_size_idx][fill_method_idx][jj] += train_time/test_repeat
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
        # add_confusion_matrix(max0,max1)
        if(max0!=max1):
            error += 1
    return "errores: %d / %d : %f%%acierto"%(error,total,100*(1-float(error)/total))


start_time = time.time()
for ii in range(test_repeat):
    single_test(sample_size,"interpolation")
print(time.time()-start_time,"s.")
print h_u_l.keys()

