"""
Yuxing Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
March 2020

THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

#from __future__ import print_function
#from __future__ import division
import os
import argparse
import distutils.util
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
# from sklearn.metrics import roc_auc_score
# from PIL import Image
# import time

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
import time
import copy
# from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from CXR_Data_Generator import DataGenerator
from my_utils import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch NIH-CXR Testing')

parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet121')
parser.add_argument('--img_size', '-sz', default=256, type=int)
parser.add_argument('--crop_size', '-cs', default=224, type=int)
parser.add_argument('--epoch', '-ep', default=50, type=int)
parser.add_argument('--batch_size', '-bs', default=64, type=int)
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float)
parser.add_argument('--gpu_id', '-gpu', default=0, type=int)

def run_test(dset, tset, results_save_path, weights_save_path):
	
	global args
	args = parser.parse_args()

	# image folder location
	img_dir = Path.get_path(tset)

	# dataset split files
	split_file_dir = './dataset_split'
	split_name = 'test'
	splits = [split_name]

	split_file_suffix = '_' + tset +'.txt' 

	split_files = {}
	for split in splits:
		split_files[split] = os.path.join(split_file_dir, 
			split+split_file_suffix)


	model = models.__dict__[args.arch](pretrained=True)
	torch.cuda.set_device(args.gpu_id)
	# number of classes
	numClass = 1
	# modify the last FC layer to number of classes
	num_ftrs = model.classifier.in_features
	model.classifier = nn.Linear(num_ftrs, numClass)
	model = model.cuda()

	model_path = weights_save_path +  dset + '/'+ args.arch+'_img_size_'+str(args.img_size)+'_crop_size_'+ str(args.crop_size) + '_batch_size_' + str(args.batch_size)+'_lr_'+str(0.001) +'_epoch_'+str(Path.get_epoch_no(dset))
	model.load_state_dict(torch.load(model_path)['state_dict'])

	return test(results_save_path,dset,tset, img_dir, split_files[split_name], split_name, model, batch_size=args.batch_size, \
		img_size=args.img_size, crop_size=args.crop_size, gpu_id=args.gpu_id)


def test(results_save_path, dset,tset,img_dir, split_test, split_name, model, batch_size, img_size, crop_size, gpu_id):

	# -------------------- SETTINGS: CXR DATA TRANSFORMS -------------------
	normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
	data_transforms = {split_name: transforms.Compose([
		transforms.Resize(img_size),
		# transforms.RandomResizedCrop(crop_size),
		transforms.CenterCrop(crop_size),
		transforms.ToTensor(),
		transforms.Normalize(normalizer[0], normalizer[1])])}

	# -------------------- SETTINGS: DATASET BUILDERS -------------------
	datasetTest = DataGenerator(img_dir=img_dir, split_file=split_test,
								transform=data_transforms[split_name])
	dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size,
								shuffle=False, num_workers=8, pin_memory=True)

	dataloaders = {}
	dataloaders[split_name] = dataLoaderTest

	print('Number of testing CXR images: {}'.format(len(datasetTest)))
	dataset_sizes = {split_name: len(datasetTest)}
 
	# -------------------- TESTING -------------------
	model.eval()
	running_corrects = 0
	output_list = []
	label_list = []
	preds_list = []

	with torch.no_grad():
		# Iterate over data.
		for data in dataloaders[split_name]:
			inputs, labels, img_names = data

			labels_auc = labels
			labels_print = labels
			labels_auc = labels_auc.type(torch.FloatTensor)
			labels = labels.type(torch.LongTensor) #add for BCE loss
			
			# wrap them in Variable
			inputs = inputs.cuda(gpu_id, non_blocking=True)
			labels = labels.cuda(gpu_id, non_blocking=True)
			labels_auc = labels_auc.cuda(gpu_id, non_blocking=True)

			labels = labels.view(labels.size()[0],-1) #add for BCE loss
			labels_auc = labels_auc.view(labels_auc.size()[0],-1) #add for BCE loss
			# forward
			outputs = model(inputs)
			# _, preds = torch.max(outputs.data, 1)
			score = torch.sigmoid(outputs)
			score_np = score.data.cpu().numpy()
			preds = score>0.5
			preds_np = preds.data.cpu().numpy()
			preds = preds.type(torch.cuda.LongTensor)

			labels_auc = labels_auc.data.cpu().numpy()
			outputs = outputs.data.cpu().numpy()

			#lav Image Name Printing
			#for j in range(len(img_names)):
			#	print(str(img_names[j]) + ': ' + str(score_np[j]) + ' GT: ' + str(labels_print[j]))

			for i in range(outputs.shape[0]):
				output_list.append(outputs[i].tolist())
				label_list.append(labels_auc[i].tolist())
				preds_list.append(preds_np[i].tolist())

			# running_corrects += torch.sum(preds == labels.data)
			# labels = labels.type(torch.cuda.FloatTensor)
			running_corrects += torch.sum(preds.data == labels.data) #add for BCE loss

	acc = np.float(running_corrects) / dataset_sizes[split_name]
	auc = metrics.roc_auc_score(np.array(label_list), np.array(output_list), average=None)
	# print(auc)
	fpr, tpr, _ = metrics.roc_curve(np.array(label_list), np.array(output_list))
	roc_auc = metrics.auc(fpr, tpr)

	ap = metrics.average_precision_score(np.array(label_list), np.array(output_list))
	
	tn, fp, fn, tp = metrics.confusion_matrix(label_list, preds_list).ravel()

	recall = tp/(tp+fn)
	precision = tp/(tp+fp)
	f1 = 2*precision*recall/(precision+recall)
	sensitivity = recall
	specificity = tn/(tn+fp)
	PPV = tp/(tp+fp)
	NPV = tn/(tn+fn)
	print('Test Accuracy: {0:.4f}  Test AUC: {1:.4f}  Test_AP: {2:.4f}'.format(acc, auc, ap))
	print('TP: {0:}  FP: {1:}  TN: {2:}  FN: {3:}'.format(tp, fp, tn, fn))
	print('Sensitivity: {0:.4f}  Specificity: {1:.4f}'.format(sensitivity, specificity))
	print('Precision: {0:.2f}%  Recall: {1:.2f}%  F1: {2:.4f}'.format(precision*100, recall*100, f1))
	print('PPV: {0:.4f}  NPV: {1:.4f}'.format(PPV, NPV))

	metrics_dict = {"train_dataset" :dset, "test_dataset" :tset,  "accuracy": acc,  "auc" : auc, "avg_precision" :ap, 
	"true_positive" : tp, "false_positive" :fp,"true_negative" :tn, "false_negative" :fn , "sensitivity" : sensitivity, "specificity" : specificity,
	 "precision" :precision, "recall": recall, "f1":f1, "PPV":PPV, "NPV" :NPV}
	
	
	# Plot all ROC curves
	plt.figure()
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve of abnormal/normal classification: '+args.arch)
	plt.legend(loc="lower right")
	plt.savefig(results_save_path + 'ROC_'+args.arch+'_'+dset+'_test_'+tset+'.pdf', bbox_inches='tight')
	plt.show()

	return metrics_dict

if __name__ == '__main__':

	train_dataset = ['bimcv2', 'rafael', 'covidx-cxr2', 'u-3-filter']
	test_set = ['bimcv2', 'rafael', 'covidx-cxr2', 'u-3-filter','xcat_covid_siemens', 'xcat_covid_carestream' ]

	results_save_path = '/Local/experiments/covid_project/results/' 
	weights_save_path = '/Local/experiments/covid_project/model_weights/' 
	results_dict = {"train_dataset" :[], "test_dataset" :[],  "accuracy": [],  "auc" : [], "avg_precision" :[], 
	"true_positive" : [], "false_positive" :[],"true_negative" :[], "false_negative" :[] , "sensitivity" : [], "specificity" : [],
	"precision" :[], "recall": [], "f1":[], "PPV":[], "NPV" :[]}
	
	for dset in train_dataset:
		for tset in test_set:
			
			metrics_dict = run_test(dset, tset, results_save_path, weights_save_path)

			for key in metrics_dict:
				results_dict[key].append(metrics_dict[key])
	
	df_results =pd.DataFrame.from_dict(results_dict)
	df_results.to_csv('/Local/experiments/covid_project/covid_project_results_metrics.csv')
