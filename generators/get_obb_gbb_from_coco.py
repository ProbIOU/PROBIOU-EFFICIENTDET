#
#  Compare masks produced by BBs, axis-aligned ellipses and full ellipses
#  with the GT maks. Also explores OBBs from segmentation masks

#
#  IMPORTANT: ignores segmentation masks that formed by mopre than one connected component
#
#

import cv2, os, pickle
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import scipy.stats as st
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from glob import glob
from sklearn.datasets.samples_generator import make_blobs
from pdb import set_trace as pause
from math import atan2

##from obb import segmentations_to_rotation_bboxs_jung, rotation_bboxs_to_poly, from_poly_to_binary
#from obb_jeffri import segmentation_to_rotation_bbox, poly_to_binary

from matplotlib.patches import Ellipse
#import matplotlib.transforms as transforms
import pandas as pd
import seaborn as sns


def segmentation_to_rotation_bbox(ann):
    """
        Format an coco annotation to a OBB

        Input: annotations extracted with cocoAPI, list of dictionaries

        Output: bboxs       -> array of number de (annotations,5) with (x,y,w,h,angle)
                list_point  -> list of multiple (2,4) matrix, that denote corners of the object

    """
    bbox      = np.zeros(5)
    p = ann
    box = np.array(p['segmentation'])
    box = np.int0(box)
    box = box.reshape([-1, 2])
    rect1  = cv2.minAreaRect(box) # ((x,y), (w,h), angle) in opencv style
    points = np.int0(cv2.boxPoints(rect1)) # 2x4 matrix, representing corner points
    bbox = [rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]]
    return bbox, points


def covariance_from_rectangle(img, tlx = 120, tly = 50, brx = 180 , bry = 100, shape = (256, 256, 3)):
	gt_bb = img.copy()
	mm = np.max(gt_bb)
	cv2.rectangle(gt_bb,(tlx, tly), (brx, bry), color = (mm, mm, mm), thickness =  -1)
#	gt_bb = gt_bb # image with rectangle
	
	#
	# Computes covariance
	#
	
	for i in range(1, mm+2):
	
		y,x = np.where(gt_bb[:,:,0] == i)
		covariance = np.cov(y,x)
		eigenvalues, eigenvectors = np.linalg.eig(covariance)
			
	
		# print instance pixels as white and convert the mask to RGB.
			
		# get the orientation of the elipse in degrees
		angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) * 180 / np.pi
	#	cv2.circle(mask_inst, cntr, 3, (0, 0, 255), -1)
	#	return gt_bb
	
	
		scale = np.sqrt(12/np.pi)
		w = scale*np.sqrt(abs(eigenvalues[1]) +1e-10 )
		h = scale*np.sqrt(abs(eigenvalues[0]) +1e-10 )
	
		x_mean = int(np.floor(x.mean()))
		y_mean = int(np.floor(y.mean()))
		cntr   = (x_mean, y_mean)
	
		cv2.ellipse(gt_bb, cntr, (int(w),int(h)), int(angle), 0, 360, (0, 255, 0), thickness = 2)
	return gt_bb


	
def iou_images(im1, im2):
	#
	#  Inputs are float images with non-zero elements denoting binary values
	#
	
	# binarize
	b1 = (im1 > 0)
	b2 = (im2 > 0)
	inter = np.sum(b1*b2)
	union = np.sum(b1+b2)
	return inter/union


#def mask_to_obb(binary_mask):
#	bbox = segmentations_to_rotation_bboxs_jung(binary_mask)
#	poly = rotation_bboxs_to_poly(bbox)
#	img = np.zeros(binary_mask.shape).astype('uint8')
#	from_poly_to_binary(img, poly)
#	return img


def process_annotations(annotation,  h_img, w_img, showResults = False):

	#
	#  Computes OBB and GBB representations from COCO annotations
	#  h_img, w_img are height and width of target image
	#  labels


	#
	#  Generates labeled image with coco annotations
	#
	seg_mask, labels = _gen_seg_mask(annotation, h_img, w_img)
	
	obb_all = []
	gbb_all = []
	final_labels = []
	
	#
	#  Scans all labels -- must use only those 20 (or 19) labels related to VOC 
	#
	MinPixels = 20	 # ignores regions that are very small -- must check!
	for i in range(1, seg_mask.max() + 1):
		if np.sum(seg_mask == i) < MinPixels:
			continue
#		label = labels[i - 1] - 1  # current category label
		mask_inst  = seg_mask.copy()
		#
		#   Generates binary image with current object
		#
		mask_inst[mask_inst!=i] = 0
		
		#
		#  Computes OBB
		#
		if ( len(annotation[i-1]['segmentation'])) > 1: # if segmentatiuon mask presents more than one component, skips
			continue
	
		obb, points = segmentation_to_rotation_bbox(annotation[i-1])
		obb_all.append(obb)
#		print(obb)
		final_labels.append(labels[i - 1])

		#select the indexes x,y form pixels that are valid
		y,x = np.where(mask_inst==i)

		if x.shape[0] == 0 or y.shape[0] == 0:
			continue

		# get the center of the segment using the mean
		x_mean = int(np.floor(x.mean()))
		y_mean = int(np.floor(y.mean()))
		
		#
		# get the covariance matrix and its related eigen values and vectors.
		#
		covariance = np.cov(y,x)
		gbb = np.array([x_mean, y_mean, covariance[0,0], covariance[1,1], covariance[0,1]])
		gbb_all.append(gbb)


		

		# print instance pixels as white and convert the mask to RGB.
#		mask_inst[mask_inst == i] = 255
#		mask_inst = cv2.cvtColor(mask_inst,cv2.COLOR_GRAY2RGB)

#		gt_mask = mask_inst[:,:,0]

		#		
		# get the orientation of the elipse in degrees
		#
		if showResults:

			eigenvalues, eigenvectors = np.linalg.eig(covariance)
			
			angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) * 180 / np.pi
			scale = np.sqrt(12/np.pi)  # scaling factor as in paper
	
			cntr   = (x_mean, y_mean)
	
			w = scale*sqrt(abs(eigenvalues[1]) +1e-10 )
			h = scale*sqrt(abs(eigenvalues[0]) +1e-10 )
			mask_inst[mask_inst == i] = 255
			mask_inst = cv2.cvtColor(mask_inst,cv2.COLOR_GRAY2RGB)
	#
			cv2.ellipse(mask_inst, cntr, (int(w),int(h)), int(angle), 0, 360, (255, 0, 0), thickness = 2)		
			cv2.polylines(mask_inst,[points],True,(0,0,255), thickness = 2)

			cv2.imshow('Gauss', mask_inst)
			cv2.waitKey()


		#
		#  masks
		#
#

#		gt_gaus = gt_gaus[:,:,0]
	return obb_all, gbb_all, final_labels

		



def save_object(obj, file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def _gen_seg_mask(target, h, w):
       mask = np.zeros((h, w), dtype=np.uint8)
       
       cont = 1
       labels = []

       for instance in target:
           rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
           m = coco_mask.decode(rle)
           cat = instance['category_id']
           labels.append(cat)
           
           c = cont
           if len(m.shape) < 3:
               mask[:, :] += (mask == 0) * (m * c)
           else:
               mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
           cont += 1
       return mask, labels




if __name__ == '__main__':
	
#	imgs = glob('D:/Dados/Coco2017/train2017/*.jpg')
	ann_file = 'D:/Dados/Coco2017/annotations/instances_train2017.json'
	coco = COCO(ann_file)
	count = 0
	
		
	for index in coco.imgs:
		count += 1
#		img_id = coco.imgs[index]['file_name'].replace('.jpg', '')
#		if count%100 == 0:
#			print('processing file %d of %d'%(count, len(imgs)))
#		img_id = int(img_path.split('\\')[-1].replace('.jpg', ''))
		img_metadata = coco.loadImgs(index)[0]
		cocotarget   = coco.loadAnns(coco.getAnnIds(imgIds=index))
		
			#
			# Gets segmentation_mask
			#
		
		obb, gbb, labels = process_annotations(cocotarget, img_metadata['height'], img_metadata['width'], showResults = False)
		
