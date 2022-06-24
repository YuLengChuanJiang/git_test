本实验是针对不加任何额外功能的TSN网络的训练，使用resnet101，RGB图像的实验精度比较高，而光流网络的训练比较差

# use singel ms-layer to build the flow network. 
ucf101
Flow
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt
--arch
resnet101


RGB model training 
ucf101
RGB
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_train_split_1.txt
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt
--arch
resnet101


python main.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
   --arch resnet101 --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 64 -j 4 --dropout 0.9 \
   --snapshot_pref ucf101_res_rgb_0717


python main.py ucf101 Flow /home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt\
    /home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 64 -j 4 --dropout 0.7 \
   --snapshot_pref ucf101_bninception_0714  
   
   
   python main.py ucf101 Flow /home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt\
    /home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt \
   --arch resnet101 --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 64 -j 4 --dropout 0.7 \
   --snapshot_pref ucf101_resnet101_0714  
   
   python main.py ucf101 Flow /home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt\
    /home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt \
   --arch resnet101 --num_segments 6 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 64 -j 4 --dropout 0.7 \
   --snapshot_pref ucf101_resnet101_0714  
   
   
ms-tcn   
python test_models.py ucf101 Flow /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
ucf101_resnet101_0714_flow_model_best.pth.tar --arch resnet101 --save_scores resnet_flow_score_0715

python test_models.py ucf101 Flow /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
ucf101_resnet101_0714_flow_model_best.pth.tar --arch resnet101 --save_scores resnet_flow_score_seg6_0715

python test_models.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
_rgb_model_best.pth.tar --arch BNInception --save_scores resnet_rgb_score_0715


two stream fusion
python eval_scores.py resnet_rgb_score_0715.npz resnet_flow_score_0715.npz --score_weight 1 1.5  

python main.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
--arch resnet101 --num_segments 5 \
--gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
-b 32 -j 4 --dropout 0.9 \
--snapshot_pref ucf101_seg5_0717

python main.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
--arch resnet101 --num_segments 5 \
--gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
-b 32 -j 4 --dropout 0.9 \
--snapshot_pref ucf101_seg5_0718

python main.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
--arch resnet101 --num_segments 5 \
--gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
-b 32 -j 4 --dropout 0.9 \
--snapshot_pref ucf101_seg5_071821


python main.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
--arch resnet101 --num_segments 5 \
--gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
-b 32 -j 4 --dropout 0.9 \
--snapshot_pref ucf101_seg5_0719


python test_models.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
ucf101_seg5_0718_rgb_model_best.pth.tar --arch resnet101 --save_scores resnet_rgb_score_seg5_0718


python test_models.py 
ucf101 
RGB 
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt 
ucf101_seg5_0719_rgb_model_best.pth.tar
python test_models.py 
--arch resnet101 
--save_scores score_file_0719

python eval_scores.py score_file_0719.npz resnet_flow_score_0715.npz --score_weight 1 1.5  


python main.py ucf101 Flow /home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt\
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt \
--arch resnet101 --num_segments 5 \
--gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
-b 16 -j 4 --dropout 0.9 \
--snapshot_pref ucf101_resnet101_0719  

# change dropout into 0.9
python test_models.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
ucf101_seg5_0719_rgb_model_best.pth.tar --arch resnet101 --save_scores resnet_rgb_score_seg5_0720

python eval_scores.py resnet_rgb_score_seg5_0720.npz resnet_flow_score_0715.npz --score_weight 1 1.5  

python test_models.py ucf101 Flow /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
ucf101_resnet101_0719_flow_model_best.pth.tar --arch resnet101 --save_scores resnet_flow_score_seg5_0721

ucf101 
Flow 
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt 
ucf101_resnet101_0719_flow_model_best.pth.tar 
--arch resnet101 
--save_scores 
resnet_flow_score_seg5_0721



python eval_scores.py resnet_rgb_score_seg5_0720.npz resnet_flow_score_seg5_0721.npz --score_weight 1 1.5 

python test_models.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
ucf101_seg5_0719_rgb_model_best.pth.tar --arch resnet101 --save_scores resnet_rgb_score_seg5_0722 

python test_models.py ucf101 Flow /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
ucf101_seg5_0723_rgb_model_best.pth.tar --arch resnet101 --save_scores resnet_flow_score_seg5_0726 


ucf101
Flow
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_train_split_1.txt
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt
--arch
resnet101


python test_models.py ucf101 RGB /home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
ucf101_seg5_0723_rgb_model_best.pth.tar --arch resnet101 --save_scores split1_rgb_seg5_0722 

ucf101
RGB
/home/ange/projects/temporal-segment-networks/data/ucf101_rgb_val_split_1.txt
ucf101_seg5_0723_rgb_model_best.pth.tar
--arch
resnet101

ucf101
Flow
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt
ucf101_seg5_0723_flow_model_best.pth.tar
--arch
resnet101

# 20220119 
ucf101
Flow
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt
--arch
resnet101
--num_segments
3
--gd
20
--lr
0.001
--lr_steps
190
300
--epochs
340
-b
64
-j
8
--dropout
0.8
--snapshot_pref
ucf101_res_s3_220118
--save_model_path
/home/ange/project-2022/model-0118/
--resume
/home/ange/project-2022/model-0118/ucf101_res_s3_220118_flow_checkpoint.pth.tar
--eval-freq
1
--print-freq
20


ucf101
RGB
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt
--arch
resnet101
--num_segments
3
--gd
20
--lr
0.001
--lr_steps
30
60
--epochs
80
-b
128
-j
8
--dropout
0.8
--snapshot_pref
ucf101_res_s3_220119
--save_model_path
/home/ange/project-2022/model-0118/
--resume
/home/ange/project-2022/model-0118/ucf101_res_s3_220118_flow_checkpoint.pth.tar
--eval-freq
1
--print-freq
20


ucf101
RGB
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt
--arch
resnet101
--num_segments
3
--gd
20
--lr
0.001
--lr_steps
30
60
--epochs
80
-b
128
-j
8
--dropout
0.8
--snapshot_pref
ucf101_res_s3_220119
--save_model_path
/home/ange/project-2022/model-0118/
--resume
''
--eval-freq
1
--print-freq
20


ucf101
Flow
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt
--arch
resnet101
--num_segments
3
--gd
20
--lr
0.001
--lr_steps
190
300
--epochs
340
-b
64
-j
8
--dropout
0.8
--snapshot_pref
ucf101_res_s3_220118
--save_model_path
/home/ange/project-2022/model-0118/
--resume
/home/ange/project-2022/model-0118/ucf101_res_s3_220118_flow_checkpoint.pth.tar
--eval-freq
1
--print-freq
20



# 对segments为5的模型进行训练 2022012012
ucf101
Flow
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_train_split_1.txt
/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt
--arch
resnet101
--num_segments
5
--gd
20
--lr
0.001
--lr_steps
190
300
--epochs
340
-b
64
-j
8
--dropout
0.8
--snapshot_pref
ucf101_res_s5_220120
--save_model_path
/home/ange/project-2022/model-0120/
--resume
/home/ange/project-2022/model-0120/ucf101_res_s5_220120_flow_checkpoint.pth.tar
--eval-freq
5
--print-freq
20


# 对segments为3的反向光流模型进行训练 2022012020,模型的初始化采用的是正向光流模型的初始化参数
ucf101
Flow
/home/ange/projects/temporal-segment-networks/data/back_new_ucf101_flow_train_split_1.txt
/home/ange/projects/temporal-segment-networks/data/back_new_ucf101_flow_val_split_1.txt
--arch
resnet101
--num_segments
3
--gd
20
--lr
0.001
--lr_steps
190
300
--epochs
340
-b
64
-j
8
--dropout
0.8
--snapshot_pref
ucf101_back_res_s3_220120
--save_model_path
/home/ange/project-2022/back-flow-3-model-0120/
--resume
/home/ange/project-2022/model-0118/ucf101_res_s3_220118_flow_model_best.pth.tar
--eval-freq
2
--print-freq
20
--scratch True