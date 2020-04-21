from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector
from utils.image import get_affine_transform, affine_transform
import pycocotools.mask as rletools

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  #opt.debug = max(opt.debug, 1)
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    out = None
    out_name = opt.demo[opt.demo.rfind('/') + 1:]
    if opt.save_video:
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      out = cv2.VideoWriter('../results/{}.mp4'.format(
        opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
          opt.video_w, opt.video_h))
    detector.pause = False
    cnt = 0
    results = {}
    if opt.load_results != '':
      load_results = json.load(open(opt.load_results, 'r'))
    while True:
        cnt += 1
        _, img = cam.read()
        if opt.resize_video:
          try:
            img = cv2.resize(img, (opt.video_w, opt.video_h))
          except:
            print('FINISH!')
            save_and_exit(opt, out, results, out_name)
        if cnt < opt.skip_first:
          continue
        try:
          cv2.imshow('input', img)
        except:
          print('FINISH!')
          save_and_exit(opt, out, results, out_name)
        input_meta = {'pre_dets': []}
        img_id_str = '{}'.format(cnt)
        if opt.load_results:
          input_meta['cur_dets'] = load_results[img_id_str] \
            if img_id_str in load_results else []
          if cnt == 1:
            input_meta['pre_dets'] = load_results[img_id_str] \
              if img_id_str in load_results else []
        ret = detector.run(img, input_meta)
        time_str = 'frame {} |'.format(cnt)
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        results[cnt] = ret['results']
        print(time_str)
        if opt.save_video:
          out.write(ret['generic'])
        if cv2.waitKey(1) == 27:
          print('EXIT!')
          save_and_exit(opt, out, results, out_name)
          return  # esc to quit
    save_and_exit(opt, out, results)
  else:
    # Demo on images, currently does not support tracking
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    #result_txt = open('/home/zhe.zhao/' + opt.demo.split('/')[-1] + '.txt','w')
    print('image num ', len(image_names))
    cnt = -1
    for (image_name) in image_names:
      cnt += 1
      #if cnt > 2 :
      #  break
      input_meta = {'pre_dets': []}
      ret = detector.run(image_name, input_meta)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
      image = cv2.imread(image_name)

      trans = get_affine_transform(ret['meta']['c'], ret['meta']['s'], 0, ( ret['meta']['out_width'], ret['meta']['out_height']), inv=1)
      results = ret['results']
      mask = np.zeros(image.shape, dtype=np.uint16)

      for j in range(len(results)):
        if results[j]['score'] > opt.vis_thresh:
          item = results[j]
          #### track seg
          if 'seg' in item:
            segment = cv2.warpAffine(item['seg'], trans, (image.shape[1],image.shape[0]),
                                         flags=cv2.INTER_CUBIC)
            bbox = item['bbox']
            w,h = bbox[2:4] - bbox[:2]
            ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

            segment_mask = np.zeros_like(segment)
            pad_rate = 0.1
            x, y = np.clip([ct[0] - (1 + pad_rate) * w / 2, ct[0] + (1 + pad_rate) * w / 2], 0,
                           segment.shape[1] - 1).astype(np.int), \
                   np.clip([ct[1] - (1 + pad_rate) * h / 2, ct[1] + (1 + pad_rate) * h / 2], 0,
                           segment.shape[0] - 1).astype(np.int)
            segment_mask[y[0]:y[1], x[0]:x[1]] = 1
            segment = segment_mask*segment
            
            #mask = np.zeros(image.shape, dtype=np.uint8, order="F")
            mask[segment>0] = item['tracking_id'] + item['class']*1000# 1
            #rle = rletools.encode(mask)
            #rle = rletools.merge(rle)
            #print(rle)
          #track_id = item['tracking_id'] + item['class']*1000
          #result_txt.write(str(cnt)+ ' '+ str(track_id) + ' '+ str(item['class']) + ' '+ str(375) + ' '+ str(1242) +' '+rle['counts'].decode(encoding='UTF-8')+'\n')
      cv2.imwrite('/home/zhe.zhao/packages/mots_tools/test_images/'+ opt.demo.split('/')[-1] + '/%06d.png'%cnt,mask)

def save_and_exit(opt, out=None, results=None, out_name=''):
  print('results')
  if opt.save_results and (results is not None):
    save_dir =  '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))
  if opt.save_video and out is not None:
    out.release()
  import sys
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
