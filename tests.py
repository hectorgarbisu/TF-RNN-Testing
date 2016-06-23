__author__ = 'geco'
import RNN
import numpy as np
from dataset_loader import dataset_loader
import tensorflow as tf

dataset_path = "./dataset/"

##################################
######## FIXED PARAMETERS ########
##################################

num_epochs = 2000
alpha = 0.01
batch_size = 100
nW_hidden = 20
test_repeat = 10
##################################
########### VARIABLES ############
##################################

"sample sizes' = [int(100*(1.3**ii)) for ii in range(10)]"
# sss = [100, 130, 169, 219, 285, 371, 482, 627, 815, 1060]
# "num steps' (% of sample_size; 0% = 1 single step)"
# nss = [0, 1, 1.5, 3,  6, 15, 30, 50, 70, 85]

sss = [100]
nss = [1]

def single_test(sample_size,percentage_steps,ii):
    num_steps = (percentage_steps//100)*sample_size
    if(num_steps<1):
        num_steps = 1
    input_size = 2*(sample_size-num_steps)
    dl = dataset_loader(dataset_path)
    dl.load(fixed_sig_size=sample_size)
    la_to_ho = dl.get_labels_to_hot_dict()
    num_classes = len(la_to_ho)
    rnn = RNN.RNN(input_size,nW_hidden,num_steps,num_classes)
    # Get batch and its labels
    batch,labels,hotone_labels = dl.next_2d_batch(batch_size)
    train(rnn,dl)
    test(rnn,dl,la_to_ho)
##################
##### TRAIN ######
##################
def train(rnn,dl):
    err = 0
    for ii in range(num_epochs):
        batch,_,hotone_labels = dl.next_2d_batch(batch_size)
        rnn.feed_batch(batch,hotone_labels)
        err = rnn.error(batch,hotone_labels)
        if (ii % (num_epochs//10)) == 0:
            print "error medio:",err/(ii+1)

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
        if(max0!=max1):
            error += 1
    print "errores: ",error, "/",total," : ",100*(1-float(error)/total),"% acierto"


for sample_size in sss:
    for num_steps in nss:
        for ii in range(test_repeat):
            single_test(sample_size,num_steps,ii)

