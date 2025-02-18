python test_widerface.py --trained_model weights/train_rotate_nonface2/mobilenet0.25_Final.pth --network mobile0.25 --save_folder results/mnet0.25_rotate_nonface-wider_val_txt
cd widerface_evaluate/
# python setup.py build_ext --inplace
python evaluation.py -p ../results/mnet0.25_rotate_nonface-wider_val_txt

python test_fddb.py --trained_model weights/train_rotate_nonface2/mobilenet0.25_Final.pth --network mobile0.25 --save_folder results/mnet0.25_rotate_nonface-fddb_txt
cd FDDB_Evaluation
# python setup.py build_ext --inplace
python evaluate.py -p  /data/lijw/code/Retinaface_pytorch/results/mnet0.25_rotate_nonface-fddb_txt/

# nonface testset
python test_widerface.py --trained_model weights/train_rotate_nonface2/mobilenet0.25_Final.pth --network mobile0.25 --save_folder results/mnet0.25_rotate_nonface-non_face_txt --dataset_folder ./data/non_face/images/
cd utils
python face_stat.py
