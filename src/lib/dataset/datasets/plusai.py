from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math

from ..generic_dataset import GenericDataset
from utils.ddd_utils import compute_box_3d, project_to_image

class PlusAI(GenericDataset):
  num_categories = 8
  default_resolution = [768, 1024]
  # ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
  #       'Tram', 'Misc', 'DontCare']
  class_name = ['car','bus','truck','moto','person','bike','cone','barri']
  # negative id is for "not as negative sample for abs(id)".
  # 0 for ignore losses for all categories in the bounding box region
  cat_ids = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8}
  max_objs = 200
  def __init__(self, opt, split):

    img_dir = '/data/tracking_data'
    ann_path = '/data/tracking_data/tracking_coco.json'
    self.images = None
    # load image list and coco
    super(PlusAI, self).__init__(opt, split, ann_path, img_dir)
    self.alpha_in_degree = False
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))


  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    pass

  def save_results(self, results, save_dir):
    pass

  def run_eval(self, results, save_dir):
    # import pdb; pdb.set_trace()
    pass