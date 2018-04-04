import sys
import argparse
import numpy as np
from sklearn.decomposition import PCA

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def prn_shape(arr,nm):
	print ("The shape of ",str(nm)," is: ",arr.shape)

parser = argparse.ArgumentParser()
parser.add_argument("K", help="K", type=int)
parser.add_argument("D", help="D, the dimensionality target for scikit-PCA", type=int)
parser.add_argument("N", help="N < 1000, the number of test images to select", type=int)
parser.add_argument("PATH_TO_DATA", help="PATH_TO_DATA, path to CIFAR-10 dataset python version file.")
args = parser.parse_args()

data = unpickle(args.PATH_TO_DATA)
#labels_byte_string = bytes('labels','utf-8')
#image_dataset = data[labels_byte_string]
image_data = data[b'data'][:1000]
#print(image_data.shape)
#print(len(image_data))
label_data = data[b'labels'][:1000]
#print(len(label_data))

#print(type(image_data))

final_image_data = np.ndarray(shape=(len(image_data),1024))
for i in range(0,len(image_data)):
	image = image_data[i]
	r_channel_component = 0.299*image[:1024]
	g_channel_component = 0.587*image[1024:2048]
	b_channel_component = 0.114*image[2048:]
	grayscale = r_channel_component+g_channel_component+b_channel_component
	final_image_data[i] = grayscale
	
train_images = final_image_data[args.N:]
#prn_shape(train_images,"train_images")

train_labels = label_data[args.N:]
#print(len(train_labels))

test_images = final_image_data[:args.N]
#prn_shape(test_images,"test_images")

test_labels = label_data[:args.N]
#print(len(test_labels))

#remember to try one with standardised data. transform test data to pca first before KNN.
pca = PCA(n_components=args.D, svd_solver= 'full')
pca.fit(train_images)
train_reduced = pca.transform(train_images)
#	prn_shape(train_reduced,"train_reduced")
test_reduced = pca.transform(test_images)
#test_reduced = test_images
#prn_shape(test_reduced,"test_reduced")
predicted_labels = np.ndarray(shape=(len(test_labels),1),dtype=int)
#prn_shape(predicted_labels,"predicted_labels")

for i in range(0,len(test_labels)):
	test_datapoint = test_reduced[i]
	#!find metric for all training points to vote
	test_datavector_mat = np.tile(test_datapoint,(len(train_labels),1))
		#print(test_datapoint)
		#print(test_datavector_mat)
	#!Vector difference b/w all voters and test point
	diff_mat = np.subtract(test_datavector_mat,train_reduced)
		#print(diff_mat)
	#!Square of Euclidean Distance
	metric_inverse = np.sqrt(np.sum(np.square(diff_mat),1))

		#print(metric_inverse)
	#!Metric Calculation
	from scipy.spatial import distance
	metric = np.reciprocal(metric_inverse)
	#print("metric",metric[0])
	metric_inverse_2 = np.array([(1/distance.euclidean(x,test_datapoint)) for x in train_reduced])
	#print(type(metric_inverse_2))
	#print("metric2",metric_inverse_2[0])
	metric = metric_inverse_2
	#print(list(metric)==metric_inverse_2)
		#print(metric.shape)
		#print(metric)
	#!Assign labels to each metric from training labels data	
	indexed_metric = zip(train_labels,metric)
		#print(indexed_metric)
	#!sorted labeled list according to votes
	voted_metric = sorted(indexed_metric, key=lambda x: float(x[1]), reverse=True)
		#print(voted_metric)
	#!Extract all labels sorted in decreasing order	
	k_voted_metric = voted_metric[:args.K]
	#print("k_voted_metric",k_voted_metric)
	
	total_votes = {}
	for label,weight in k_voted_metric:
		if label not in total_votes:
			total_votes[label] = weight
		else:
			total_votes[label] += weight
	
	max_vote = max(total_votes, key=total_votes.get)
	predicted_labels[i] = max_vote

with open('2428734682.txt','w') as f:
	for (x,y) in zip(predicted_labels,test_labels):
		f.write(str(x[0])+" "+str(y)+"\n")













