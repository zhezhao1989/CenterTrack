cd src
# train, the model is finetuned from a CenterNet detection model from the CenterNet model zoo.
python main.py tracking --dataset plusai --arch dlav0_34 --exp_id plusai --max_frame_dist 0.25 --resume --load_model ../exp/tracking/plusai/model_last.pth  --gpus 0 --batch_size 4 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1