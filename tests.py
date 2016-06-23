__author__ = 'geco'
import RNN
import numpy as np
from dataset_loader import dataset_loader

dataset_path = "./dataset/"
num_epochs = 2000
# dataset_lenght = 100
sample_size = 100
alpha = 0.01
nW_hidden = 20
batch_size = 30
num_steps = 20
input_size = 2*(sample_size-num_steps)
dl = dataset_loader(dataset_path)
dl.load(fixed_sig_size=sample_size)
la_to_ho = dl.get_labels_to_hot_dict()
print la_to_ho
num_classes = len(la_to_ho)

rnn = RNN.RNN(input_size,nW_hidden,num_steps,num_classes)
# Get batch and its labels
batch,labels,hotone_labels = dl.next_2d_batch(batch_size)
# Get the labels in the hot-one form, using the dictionary
# batch2,labels2 = dl.next_2d_batch(batch_size)
# Number of inputs will depend on the adjusted size of each sample
# Number of outputs will depend on the number of different classes to classify

##################
##### TRAIN ######
##################

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