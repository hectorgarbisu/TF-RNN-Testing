__author__ = 'geco'
from os import listdir
from random import shuffle
from copy import deepcopy
from math import floor,ceil
from matplotlib import pyplot
import numpy as np


class dataset_loader:


    """ path """
    def __init__ (self, path="./dataset/"):
        self.path = path
        self.training_set = []
        self.test_set = []
        self.training_labels = []
        self.test_labels = []
        self.class_cardinalities = dict()
        self.labels_to_hot = dict()
        self.batch_idx = 0

    "Whenever a class set is under a certain (min_per_label) number of samples" \
    "it will be ignored. Program will also avoid reading over (max_per_label) samples" \
    "of a given class"
    "Each data sample will be adjusted to have exactly (fixed_sig_size) points"
    def load(self, min_per_label=100, max_per_label=140, fixed_sig_size=100):
        file_list = listdir(self.path)
        "get list of available class names"
        class_cardinalities = self._get_labels(file_list)
        "get cardinality of each class (dictionary)"
        print "labels dict: ",class_cardinalities
        "Cap every class set to a maximum number of members equal" \
        "to the size of the smaller class (ccty)"
        class_cardinalities = self._filter_labels(class_cardinalities,min_per_label,max_per_label)
        print "filtered and adjusted labels: ", class_cardinalities
        "read (ccty) files of each class from folder at random order" \
        "get files, its labels, and its sizes"
        files,labels,sizes = self._get_files(file_list,class_cardinalities)
        print "files: ",len(files)," labels: ",labels," sizes: ",sizes
        "center around first point"
        centered_sigs = self._center(files)
        "adjust over signature size, not canvas size"
        resized_centered_sigs = self._resize(centered_sigs)
        # resized_centered_sigs = files
        "Fix every sample to (fixed_sig_length)"
        fixed_data = self._interpolate_points(resized_centered_sigs,sizes,fixed_sig_size)
        self.labels_to_hot = self._labels_to_hot(class_cardinalities)
        self.class_cardinalities = class_cardinalities

        "Separate training and test set"
        training_indexes,test_indexes = self._get_training_and_test_indexes(labels)
        # print len(self.training_set),len(self.training_labels),len(self.test_set),len(self.test_labels)
        for ii in training_indexes:
            self.training_set.append(fixed_data[ii])
            self.training_labels.append(labels[ii])
        for ii in test_indexes:
            self.test_set.append(fixed_data[ii])
            self.test_labels.append(labels[ii])
        # print len(self.training_set),len(self.training_labels),len(self.test_set),len(self.test_labels)
        # print [(labels[i],", prev size: ",len(files[i])," fixed size:",len(fixed_data[i])) for i in range(len(files))]

        _, (b,c) = pyplot.subplots(2)
        # b.plot([p[0] for p in files[4]],[p[1] for p in files[4]],'.')
        # c.plot([p[0] for p in fixed_data[4]],[p[1] for p in fixed_data[4]],'.')
        # # c.plot(fixed_data[4][::][0],fixed_data[4][::][1])
        # pyplot.show()

    def _get_labels(self,filelist):
        labels_dict = dict()
        for filename in filelist:
            label = self._label_from_filename(filename)
            try:
                labels_dict[label] = labels_dict[label]+1
            except:
                labels_dict[label] = 1

        return labels_dict

    def _label_from_filename(self,filename):
        return str.split(filename,"-")[0]

    def _filter_labels(self,labels_dict,min_per_label,max_per_label):
        labels_dict_copy = labels_dict.copy()
        min_val = max_per_label
        for label,ctty in labels_dict_copy.iteritems():
            if (ctty<min_per_label):
                "If a class doesn't have enough members it will be ignored"
                # print "class ",label,"is deleted because: ",ctty,"<",min_per_label
                labels_dict.pop(label)
            else:
                min_val = min(min_val,ctty)
        for label,ctty in labels_dict.iteritems():
            "If a class does have too many members it will be cut"
            # print "class ",label,"is capped from ",ctty,"to ",min_val
            labels_dict[label]=min_val
        return labels_dict

    def _get_files(self,file_list,labels_dict):
        """ Return a vector of "files", a vector with its labels and a vector of sizes """
        shuffle(file_list)
        files,labels,sizes = [[],[],[]]
        for filename in file_list:
            label = self._label_from_filename(filename)
            if (label in labels_dict):
                " open and read the file " \
                " must check that label belongs in labels_dict"
                sample_points, period, size = self._read_points(filename)
                if(size>10):
                    files.append(sample_points)
                    " label is inserted as is "
                    labels.append(label)
                    " after opening the file, see how many points it contains"
                    sizes.append(size)
        return files,labels,sizes

    def _read_points(self,filename):
        f = open(self.path+"/"+filename,'r')
        period = f.readline()
        xdim, ydim = f.readline().split()
        _ = f.readline()
        data = list()
        for line in f:
            xi,yi = line.split()
            "coordinates between -1 and 1"
            data.append((2*float(xi)/float(xdim)-1,1-2*float(yi)/float(ydim)))
        dataset_length = len(data)
        f.close()
        return data,period,dataset_length

    def _center(self, irregular_signatures):
        centered_irregular_signatures = []
        for irregular_signature in irregular_signatures:
            x0 = irregular_signature[0][0]
            y0 = irregular_signature[0][1]
            centered_irregular_signatures.append(
                [(xx-x0,yy-y0) for (xx,yy) in irregular_signature]
            )
        # print centered_irregular_signatures
        return centered_irregular_signatures

    def _resize(self,irregular_centered_signatures, preserve_aspect_ratio = True):
        resized_centered_signatures = []
        a0 = 0.000001 # almost_zero : monstrous thing to prevent divisions by 0
        for i_c_s in irregular_centered_signatures:
            xmax = max([xy[0]+a0 for xy in i_c_s])
            ymax = max([xy[1]+a0 for xy in i_c_s])
            xmin = min([xy[0]-a0 for xy in i_c_s])
            ymin = min([xy[1]-a0 for xy in i_c_s])
            "Fitting is made so the first point keeps at 0,0"
            xsf = min(-1/xmin,1/xmax) # x_scalation_factor
            ysf = min(-1/ymin,1/ymax) # y_scalation_factor
            if preserve_aspect_ratio:
                xsf = ysf = min(xsf,ysf)
            resized_centered_signatures.append(
                [(xx*xsf,yy*ysf) for (xx,yy) in i_c_s]
            )
        return resized_centered_signatures


    def _interpolate_points(self,data_samples,sizes,number_of_points):
        new_points = deepcopy(data_samples)
        for ii in range(len(sizes)):
            size = sizes[ii]
            # print size, len(data_samples[ii])
            if(size>number_of_points):
                # Won't need this part by the moment
                "We need to remove some points"
                residue = number_of_points-size
                remove_distance = size//residue
                new_points[ii].remove(remove_distance-1)
            elif(size<1):
                "Null signature"
                Exception("Signature without data")
            else:
                "We need to add some points"
                "Add the middle point of the next pair of points"
                "until size=number_of_points"
                current_size = size
                jj = 0
                """ Return test set and its labels """
                while(current_size<number_of_points):
                    # print "jj: ",jj," ii: ", ii," number_of_points: ",number_of_points, " current_size: ",current_size
                    new_point = self._middle_point(new_points[ii][jj],new_points[ii][jj+1])
                    new_points[ii].insert(jj+1,new_point)
                    current_size += 1
                    " loop through the whole sample as many times as needed "
                    jj = (jj+2)%(current_size-1)
        return new_points

    def _middle_point(self,p1,p2):
        return ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)

    " Unproportional sharing "
    def _get_training_and_test_indexes(self,labels,test_share=0.3):
        train_indexes = range(int(floor(test_share*len(labels))))
        test_indexes = range(int(ceil(test_share*len(labels))-1),len(labels))
        # print train_indexes
        # print test_indexes
        return test_indexes,train_indexes

    def _labels_to_hot(self,labels_dict):
        labels_to_hot = dict()
        identity_list = np.identity(len(labels_dict)).tolist()
        labels_to_hot = {labels_dict.keys()[idx] : identity_list[idx] for idx in range(len(labels_dict))}
        return labels_to_hot



    def _l_of_l_copy(self, list_of_lists):
        " Yo dawg "
        copy = list()
        for sublist in list_of_lists:
            sublist_copy = list()
            for element in sublist:
                sublist_copy.append(deepcopy(element))
            copy.append(sublist_copy)
        return copy

    def next_2d_batch(self, batch_size=1):
        j = self.batch_idx
        inputs = []
        expected_outputs = []
        for i in range(batch_size):
            current_sample = self.training_set[(i+j)%len(self.training_set)]
            flattened_sample = np.array(current_sample).flatten()
            inputs.append(flattened_sample)
            expected_outputs.append(self.training_labels[(i+j)%len(self.training_set)])
        self.batch_idx += batch_size
        self.batch_idx %= len(self.training_set)
        this_batch_hotones = [self.labels_to_hot[label] for label in expected_outputs]
        this_batch_hotones_array = np.atleast_2d(this_batch_hotones)
        return inputs,expected_outputs, this_batch_hotones_array

    def next_batch(self, batch_size=1):
        j = self.batch_idx
        inputs = []
        expected_outputs = []
        for i in range(batch_size):
            inputs.append(self.training_set[(i+j)%len(self.training_set)])
            expected_outputs.append(self.training_labels[(i+j)%len(self.training_set)])
        self.batch_idx += batch_size
        return np.atleast_2d(inputs),expected_outputs

    def get_labels_to_hot_dict(self):
        return self.labels_to_hot


    def get_test_set(self):
        return zip(self.test_set,self.test_labels)
