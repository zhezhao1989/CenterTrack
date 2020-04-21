from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import os
import cv2

CLASS_DICT = {
                'car': 'car',
                'bus': 'bus',
                'truck': 'truck',
                'motorcycle': 'moto',
                'moto': 'moto',
                'bicycle': 'bike',
                'ped': 'pedestrian',
                'cyclist': 'bike',
                'cone': 'traffic_cone',
                'van': 'car',
                'bike': 'bike',
                'barricade': 'barricade',
                'other': 'other',
                'corn_barrier': 'traffic_cone',
                'policecar': 'car',
                'ambulance': 'car',
                'fireengine': 'car',
                'animal': 'other',
                "movabletrafficsign": "movabletrafficsign",
                "heavyequipment": "heavyequipment"
              }

LABEL_MAPPING = {
    'car': 0,
    'bus': 1,
    'truck': 2,
    'moto': 3,
    'pedestrian': 4,
    'bike': 5,
    'cyclist': 5,
    'traffic_cone': 6,
    'barricade': 7,
    'van': 0  # currently treat van the same as car
}

INVERSE_LABEL_MAPPING = {
    '0': 'car',
    '1': 'bus',
    '2': 'truck',
    '3': 'moto',
    '4': 'pedestrian',
    '5': 'bike',
    '6': 'traffic_cone',
    '7': 'barricade'
}

def LoadLabel(filename):
    json_str = ''
    try:
        with open(filename, 'r') as infile:
            json_obj = json.load(infile)
        return json_obj
    except:
        return ''

if __name__ == '__main__':

    label_file = '/home/plusai/training_data/label/tracking.json'

    label = LoadLabel(label_file)

    ret = {'images': [], 'annotations': [],
           'videos': []}
    ret['videos'].append({'id': 1, 'file_name': 'us_data'})

    times = []
    num_images = 0
    for annos in label['labeling']:

        filename = annos['filename'].split('/')[3:]
        filename = '/'.join(filename)

        name = annos['filename'].split('/')[6]
        ss = name.split('.')
        ss[0] = int(ss[0].split('_')[1])
        ss[1] = int(ss[1])/100.

        num_images += 1
        image_info = {'file_name': filename,
                      'id': num_images,
                      'video_id': annos['filename'].split('/')[4],
                      'frame_id': ss[0] + ss[1]}
        times.append(ss[0]+ss[1])
        chk = False
        for anno in annos['annotations']:
            clsname = anno['class']
            if clsname not in CLASS_DICT.keys():
                continue

            label = CLASS_DICT[clsname]
            if label not in LABEL_MAPPING:
                continue
            bbox = [anno['x'],anno['y'],anno['width'],anno['height']]

            ann = {'image_id': num_images,
                   'id': int(len(ret['annotations']) + 1),
                   'category_id': LABEL_MAPPING[label],
                   'bbox': bbox,
                   'track_id': anno['uuid']}
            ret['annotations'].append(ann)
            chk = True
        if chk:
            ret['images'].append(image_info)

    times.sort()
    print(times)
    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))

    out_path = 'tracking_coco.json'
    json.dump(ret, open(out_path, 'w'))