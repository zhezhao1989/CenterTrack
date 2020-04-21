cd src
# train, the model is finetuned from a CenterNet detection model from the CenterNet model zoo.
nohup python main.py tracking,seg --dataset kitti_tracking --exp_id kitti_seg \
 --gpus 4,5,6,7 --batch_size 16 --lr 1.25e-4 \
--num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 \
--resume --load_model ../exp/tracking,seg/kitti_seg/model/model_last.pth \
--lr_step 60,100 --num_epochs 120 \
> ~/nohup.txt &