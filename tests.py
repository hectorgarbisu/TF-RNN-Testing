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
nW_hidden = 40
test_repeat = 10
e_s = 50 # error samples

num_clases = 0
sample_size_idx = 0
num_steps_idx = 0
dl = dataset_loader(dataset_path)
dl.load(fixed_sig_size=100) #mock load
plop = dl.get_labels_to_hot_dict()
##################################
########### VARIABLES ############
##################################

"sample sizes' = [int(100*(1.3**ii)) for ii in range(10)]"
decreasing = True
if decreasing:
    ## IN REVERSE FOR TIME TESTING (comment one or other)
    sss = [1060, 815, 627, 482, 371, 285, 219, 169, 130, 100]
    nss = [85, 70, 50, 30,  15, 6, 3, 1.5, 1, 0]
else:
    sss = [100, 130, 169, 219, 285, 371, 482, 627, 815, 1060]
    "num steps' (% of sample_size; 0% = 1 single step)"
    nss = [0, 1, 1.5, 3,  6, 15, 30, 50, 70, 85]

errors = np.zeros([len(sss),len(nss),e_s])
times = np.zeros([len(sss),len(nss),e_s])
confusion_matrices = np.zeros([len(sss),len(nss),6,6])
# sss = [100]
# nss = [1]

def single_test(sample_size,percentage_steps,ii):
    num_steps = (percentage_steps//100)*sample_size
    if(num_steps<1):
        num_steps = 1
    input_size = 2*(sample_size-num_steps)
    dl = dataset_loader(dataset_path)
    dl.load(fixed_sig_size=sample_size) #mock load
    la_to_ho = plop
    # la_to_ho = dl.get_labels_to_hot_dict()
    num_classes = len(la_to_ho)
    rnn = RNN.RNN(input_size,nW_hidden,num_steps,num_classes)
    # Get batch and its labels
    batch,labels,hotone_labels = dl.next_2d_batch(batch_size)
    train(rnn,dl)
    test(rnn,dl,la_to_ho)
    rnn.sess.close()
    ops.reset_default_graph()

##################
##### TRAIN ######
##################
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
            errors[sample_size_idx][num_steps_idx][jj] += err/test_repeat
            times[sample_size_idx][num_steps_idx][jj] += train_time/test_repeat
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
    print "errores: ",error, "/",total," : ",100*(1-float(error)/total),"% acierto"


def add_confusion_matrix(max0, max1, la_to_ho=plop):
    confusion_matrices[sample_size_idx][num_steps_idx][max0][max1]+=1

print plop.keys()


def print_results_to_file():
    error_file = open("%s%d-%d-error"%(result_path,sample_size,num_steps),"w")
    for ii in range(e_s):
        time_and_error = (times[sample_size_idx][num_steps_idx][ii],
                          errors[sample_size_idx][num_steps_idx][ii])
        error_file.write("%f:%f\n"%time_and_error)
        print time_and_error
    error_file.close()
    confusion_file = open("%s%d-%d-confumat"%(result_path,sample_size,num_steps),"w")
    confusion_file.write(" ".join(plop.keys())+"\n")
    for ii in range(6):
        for jj in range(6):
            confusion_file.write(" "+str(confusion_matrices[sample_size_idx][num_steps_idx][ii][jj]))
        confusion_file.write("\n")
    confusion_file.close()
    pass


for sample_size_idx,sample_size in enumerate(sss):
    for num_steps_idx,num_steps in enumerate(nss):
        case_start_time = time.time()
        print("Case: sample_size=",sample_size," steps_%=",num_steps)
        for ii in range(test_repeat):
            single_test(sample_size,num_steps,ii)
        print(time.time()-case_start_time,"s. for case: sample_size=",sample_size," steps_%=",num_steps)
        print times[sample_size_idx][num_steps_idx]
        print confusion_matrices[sample_size_idx][num_steps_idx]
        print_results_to_file()

