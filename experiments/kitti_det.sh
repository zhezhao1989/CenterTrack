cd src
# train, the model is finetuned from a CenterNet detection model from the CenterNet model zoo.
python main.py seg --dataset kitti_tracking --exp_id kitti_seg \
 --gpus 4,5,6,7 --batch_size 16 --lr 5e-4 \
--resume --load_model ../exp/seg/kitti_seg/model_last.pth