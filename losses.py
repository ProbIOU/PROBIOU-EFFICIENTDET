"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import keras
import math
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from utils.anchors import anchors_for_shape
from layers import RegressBoxes

def focal(alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer
        
        #loss = tf.math.divide_no_nan(keras.backend.sum(cls_loss), normalizer)
        #return tf.where(tf.math.is_nan(loss), 0., loss)

    return _focal


def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def smooth_l1_quad(sigma=3.0):
    """
    Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression = tf.concat([regression[..., :4], tf.sigmoid(regression[..., 4:9])], axis=-1)
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        box_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., :4], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., :4], 2),
            regression_diff[..., :4] - 0.5 / sigma_squared
        )

        alpha_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., 4:8], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., 4:8], 2),
            regression_diff[..., 4:8] - 0.5 / sigma_squared
        )

        ratio_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., 8], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., 8], 2),
            regression_diff[..., 8] - 0.5 / sigma_squared
        )
        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())

        box_regression_loss = tf.reduce_sum(box_regression_loss) / normalizer
        alpha_regression_loss = tf.reduce_sum(alpha_regression_loss) / normalizer
        ratio_regression_loss = tf.reduce_sum(ratio_regression_loss) / normalizer

        return box_regression_loss + alpha_regression_loss + 16 * ratio_regression_loss

    return _smooth_l1

''' Probabilistic IoU '''

EPS = 1e-3

def helinger_dist(x1,y1,a1,b1, x2,y2,a2,b2, freezed=False):
    '''
    Dh = sqrt(1 - exp(-Db))
    
    Db = 1/4*((x1-x2)²/(a1+a2) + (y1-y2)²/(b1+b2))-ln2 \
    1/2*ln((a1+a2)*(b1+b2)) - 1/4*ln(a1*a2*b1*b2)
    '''
    
    if freezed:
        B1 = 1/4.*(tf.math.pow(x1-x2, 2.)/(a1+a2+EPS) + tf.math.pow(y1-y2, 2.)/(b1+b2+EPS))
        B2 = 1/2.*tf.math.log((a1+a2)*(b1+b2)+EPS)
        B3 = 1/4.*tf.math.log(a1*a2*b1*b2+EPS)
        Db = B1 + B2 - B3 - tf.math.log(2.)
    else:
        Db = tf.math.pow(x1-x2, 2.)/(2*a1+EPS) + tf.math.pow(y1-y2, 2.)/(2*b1+EPS)
        
    Db = tf.clip_by_value(Db, EPS, 100.)
    
    return tf.math.sqrt(1 - tf.math.exp(-Db) + EPS)

def get_piou_values(array):
    # xmin, ymin, xmax, ymax
    xmin = array[:,0]; ymin = array[:,1]
    xmax = array[:,2]; ymax = array[:,3]
    
    # get ProbIoU values
    x = (xmin + xmax)/2.
    y = (ymin + ymax)/2.
    a = tf.math.pow((xmax - xmin), 2.)/12.
    b = tf.math.pow((ymax - ymin), 2.)/12.
    return x, y, a, b

def calc_piou(mode, target, pred, freezed=False):
    
    l1 = helinger_dist(
                *get_piou_values(target),
                *get_piou_values(pred),
                freezed=freezed
            )
    if mode=='piou_l1':
        return l1
    
    l2 = tf.math.pow(l1, 2.)
    if mode=='piou_l2':
        return l2
    
    l3 = - tf.math.log(1. - l2 + EPS)
    if mode=='piou_l3':
        return l3
    
    # smooth probIoU
    l1_f = helinger_dist(
                *get_piou_values(target),
                *get_piou_values(pred),
                freezed=True
            )
    l1_nf = helinger_dist(
                *get_piou_values(target),
                *get_piou_values(pred),
                freezed=False
            )
    return tf.where(l1_nf>0.4, l1_f, l1_nf)
    
def calc_diou_ciou(mode, bboxes1, bboxes2):
    # xmin, ymin, xmax, ymax
    
    rows = tf.cast(tf.shape(bboxes1)[0], 'float32')
    cols = tf.cast(tf.shape(bboxes2)[0], 'float32')
    cious = tf.zeros((rows, cols), dtype='float32')
    dious = tf.zeros((rows, cols), dtype='float32')
    if rows * cols == 0:
        return cious
    exchange = False
    if rows > cols:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = tf.zeros((cols, rows), dtype='float32')
        dious = tf.zeros((cols, rows), dtype='float32')
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2.
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2.
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2.
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2.

    inter_max_xy = tf.math.minimum(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = tf.math.maximum(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = tf.math.maximum(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = tf.math.minimum(bboxes1[:, :2],bboxes2[:, :2])
    
    inter = inter_max_xy - inter_min_xy
    inter = tf.where(inter<0., 0., inter)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2. + (center_y2 - center_y1)**2.
    outer = out_max_xy - out_min_xy
    outer = tf.where(outer<0., 0., outer)
    outer_diag = (outer[:, 0] ** 2.) + (outer[:, 1] ** 2.)
    union = area1+area2-inter_area
    
    if mode=='diou':
        dious = inter_area / union - (inter_diag) / outer_diag
        dious = tf.clip_by_value(dious, -1.0, 1.0)
        
        if exchange:
            dious = tf.transpose(dious)
        return 1. - dious
    
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    v = (4. / (math.pi ** 2.)) * tf.math.pow((tf.math.atan(w2 / h2) - tf.math.atan(w1 / h1)), 2.)
    
    S = tf.stop_gradient(1. - iou)
    alpha = tf.stop_gradient(v / (S + v))
    
    cious = iou - (u + alpha * v)
    cious = tf.clip_by_value(cious, -1.0, 1.0)
    
    if exchange:
        cious = tf.transpose(cious)
    
    return 1. - cious

def iou_loss(mode, phi, weight, anchor_parameters=None, freeze_iterations=0):
    
    assert phi in range(7)
    image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
    input_size = float(image_sizes[phi])
    it = 0
    
    def _iou(y_true, y_pred):
        nonlocal it
        
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]
        
        # convert to boxes values: xmin, ymin, xmax, ymax
        anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
        anchors_input = np.expand_dims(anchors, axis=0)
        regression = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])
        regression_target = RegressBoxes(name='boxes')([anchors_input, regression_target[..., :4]])

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)
        
        if 'piou' in mode:
            loss = calc_piou(mode, regression_target, regression, freezed=freeze_iterations>it)
            it += 1
        elif mode in ('diou', 'ciou'):
            loss = calc_diou_ciou(mode, regression, regression_target)
        else:
            # requires: y_min, x_min, y_max, x_max
            xmin, ymin, xmax, ymax = tf.unstack(regression, axis=-1)
            regression = tf.stack([ymin,xmin,ymax,xmax], axis=-1)
            
            xmin, ymin, xmax, ymax = tf.unstack(regression_target, axis=-1)
            regression_target = tf.stack([ymin,xmin,ymax,xmax], axis=-1)
            
            loss = tfa.losses.GIoULoss(mode=mode, reduction=tf.keras.losses.Reduction.NONE) (regression_target, regression)
        
        return tf.cast(weight, 'float32') * loss

    return _iou