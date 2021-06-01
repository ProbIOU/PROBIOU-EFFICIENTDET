# EfficientDet with ProbIoU
This repository is a fork from the Tensorflow implementation of EfficientDet used in the "Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection"
(link TBD). It includes several losses for regressing the HBBs, such as: IoU, CIoU, DIoU, GIoU, and the proposed ProbIoU L1 & L2.

This is an implementation of [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) for object detection on Keras and Tensorflow. 
The project is based on the official implementation [google/automl](https://github.com/google/automl), [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
and the [qubvel/efficientnet](https://github.com/qubvel/efficientnet).

## About pretrained weights
* The pretrained EfficientNet weights on imagenet are downloaded from [Callidior/keras-applications/releases](https://github.com/Callidior/keras-applications/releases)
* The pretrained EfficientDet weights on PASCAL VOC 2007 will be available soon...

Thanks for their hard work.
This project is released under the Apache License. Please take their licenses into consideration too when use this project.

**Updates**
- [06/01/2021] First commit.

## Train
### build dataset 
1. Pascal VOC (for 2007+20012)
    * Download VOC2007 and VOC2012, copy all image files from VOC2007 to VOC2012.
    * Append VOC2007 train.txt to VOC2012 trainval.txt.
    * Overwrite VOC2012 val.txt by VOC2007 val.txt.
2. MSCOCO 2017
    * Download images and annotations of coco 2017
    * Copy all images into datasets/coco/images, all annotations into datasets/coco/annotations
3. Other types please refer to [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet))
### train
We recommend using the jupyter notebook [train.ipynb](https://github.com/ProbIOU/PROBIOU-EFFICIENTDET/blob/main/train.ipynb) for training your model with the parameters used on the ProbIoU paper.
## Evaluate
We recommend using the jupyter notebook [evaluate.ipynb](https://github.com/ProbIOU/PROBIOU-EFFICIENTDET/blob/main/evaluate.ipynb) for evaluting the trained models on both IoU and ProbIoU (i.e. 1-ProbIou) metrics.

## Results on PASCAL VOC 2007
| **Loss**          | **IoU50**  | **IoU75**  | **IoU50:95** | **PIoU50** | **PIoU75** | **PIoU50:95** |
| ----------------  | ---------- | ---------- | ------------ | ---------- | ---------- | ------------- |
| ProbIoU           | **72.61**  | 44.24      | 42.60        | **76.70**  | **64.15**  | **56.76**     |
| GIoU              | 70.45      | 43.96      | 42.23        | 74.26      | 62.12      | 55.33         |
| DIoU              | 70.07      | **44.74**  | 42.64        | 73.52      | 62.26      | 55.31         |
| CIoU              | 70.53      | 45.35      | **42.94**    | 74.42      | 63.04      | 55.87         |
| Smooth L1         | 70.20      | 42.02      | 40.72        | 74.09      | 61.26      | 54.49         |
