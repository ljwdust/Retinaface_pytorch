python train.py --network mobile0.25 --resume_net ./weights/official/mobilenet0.25_Final.pth --save_folder ./weights/train_rotate_nonface2/

python train.py --network mobile0.25 --lr=2e-4 --resume_net ./weights/train_rotate_nonface2/mobilenet0.25_Final.pth --save_folder ./weights/train_rotate_nonface2_960size-2/
