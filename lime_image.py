"""
Functions for explaining classifiers that use Image data.
"""
import copy
from functools import partial

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from skimage.color import gray2rgb

from . import lime_base
from .wrappers.scikit_image import SegmentationAlgorithm

import os
import cv2
import sys
import torch
import matplotlib.pyplot as plt

class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    def get_image_and_mask(self, label, positive_only=True, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that contribute to
                the prediction of the label. Otherwise, use the top
                num_features superpixels, which can be positive or negative
                towards the label
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: TODO

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = 1 if w < 0 else 2
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
                for cp in [0, 1, 2]:
                    if c == cp:
                        continue
                    # temp[segments == f, cp] *= 0.5
            return temp, mask


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, image, pytorch_img, inpaint_model, classifier_fn, l_map, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None, fill_type='LIME', num_super_pixel=50, sav_path='', target_category=0):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        try:
            segments = segmentation_fn(image)
        except ValueError as e:
            raise e

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (255*0.485, 255*0.456, 255*0.406)
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, pytorch_img, inpaint_model, fudged_image, segments,
                                        classifier_fn, num_samples, label_map=l_map,
                                        batch_size=batch_size, f_type=fill_type, num_super_pixel=num_super_pixel, save_path=sav_path, gt_category=target_category)
        # import ipdb
        # ipdb.set_trace()
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp


    def unnormalize(self, img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        preprocessed_img = img.copy()
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] * stds[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] + means[i]
        return preprocessed_img


    def data_labels(self,
                    image, pytorch_img, inpaint_model, 
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples, label_map,
                    batch_size=10, f_type='LIME', num_super_pixel=50, save_path='', gt_category=0):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        
        labels = []
        data[0, :] = 1
        imgs = []
        temp_mask = torch.tensor([]) 
        ind = 0
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            
            if f_type == 'LIME':
                # Original LIME
                temp[mask] = fudged_image[mask]
            elif f_type == 'LIMEG': 
                if temp_mask.shape[0] == 0:
                    temp_mask = (1 - torch.from_numpy(mask).unsqueeze(0).float()).expand(3, mask.shape[0], mask.shape[1]).unsqueeze(0)
                else:
                    temp_mask = torch.cat((temp_mask, (1 - torch.from_numpy(mask).unsqueeze(0).float()).expand(3, mask.shape[0], mask.shape[1]).unsqueeze(0)), dim=0)

            if f_type == 'LIME':
                # Save intermediate steps
                outputs = classifier_fn(np.array([temp]))
                amax, aind = outputs.max(dim=1)
                gt_val = outputs.data[:, gt_category]
                cv2.imwrite(
                    os.path.join(save_path, 'intermediate_{:04d}_{}_{:.3f}_{}_{:.3f}.jpg'
                                 .format(ind, label_map[aind.item()].split(',')[0].split(' ')[0].split('-')[0],
                                         amax.item(), label_map[gt_category].split(',')[0].split(' ')[0].split('-')[0],
                                         gt_val.item())), cv2.cvtColor(np.array([temp])[0, :], cv2.COLOR_BGR2RGB))
                ind += 1           
            
            imgs.append(temp)
            if len(imgs) == batch_size:
                if f_type == 'LIMEG':
                    inpaint_img, _ = inpaint_model.generate_background(pytorch_img, temp_mask, batch_process=True)
                    inpaint_img = pytorch_img.cpu() * temp_mask + inpaint_img.cpu() * (1 - temp_mask)
                    inpaint_img = np.uint8(255 * self.unnormalize(np.moveaxis(inpaint_img.cpu().detach().numpy().transpose(), 0, 1)))
                    inpaint_img = np.rollaxis(inpaint_img, -1)
                    preds = classifier_fn(inpaint_img)
                    for ii in range(batch_size):
                       temp_output = classifier_fn(np.expand_dims(inpaint_img[ii, :], axis=0))
                       # Save intermediate steps
                       amax, aind = temp_output.max(dim=1)
                       gt_val = temp_output.data[:, gt_category]
                       cv2.imwrite(os.path.join(save_path, 'intermediate_{:04d}_{}_{:.3f}_{}_{:.3f}.jpg'
                           .format(ind*batch_size + ii, label_map[aind.item()].split(',')[0].split(' ')[0].split('-')[0], 
                               amax.item(), label_map[gt_category].split(',')[0].split(' ')[0].split('-')[0], 
                               gt_val.item())), cv2.cvtColor(inpaint_img[ii, :], cv2.COLOR_BGR2RGB))

                    ind += 1
                    labels.extend(preds.data.cpu().numpy())
                    temp_mask = torch.tensor([])
                    imgs=[]
                else:
                    preds = classifier_fn(np.array(imgs))
                    labels.extend(preds.data.cpu().numpy())
                    imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds.data.cpu().numpy())
        return data, np.array(labels)
