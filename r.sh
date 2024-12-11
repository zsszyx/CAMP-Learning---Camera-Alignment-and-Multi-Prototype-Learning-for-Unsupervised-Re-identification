#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.05
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.01


#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.01
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.03
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 128 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.5
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 128 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.8
#
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 1
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.8
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.5
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.1
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.05

#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 100 -i 100 --global-momentum 0.01
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 100 -i 100 --global-momentum 0.03
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 100 -i 100 --global-momentum 0.05
#CUDA_VISIBLE_DEVICES=1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 80 -i 200 --global-momentum 0.01
#CUDA_VISIBLE_DEVICES=1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 80 -i 200 --global-momentum 0.03
# 174
#CUDA_VISIBLE_DEVICES=0 python main.py -b 256 -d market1501 --data-dir /mnt/Datasets -e 60 -i 200 --global-momentum 0.3


#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 60 -i 200 --global-momentum 0.001
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 --cams-decay 0.6 -e 60
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 80
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 80 -i 200 -s 30 55
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 20 40 60
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 -e 120 -i 200 -s 20 40 60 80 \86

#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 120 -i 150 -s 20 40 60 80
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 55 --p 0.01
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 55 --p 0.05
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 55 --p 0.5
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 55 --p 0.6
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 55 --p 0.7
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 55 --p 0.8
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 150 -s 40 80 -d msmt17
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 300 -s 40 80 -d msmt17
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 80 -i 300 -s 30 55 -d msmt17
# 87.1

# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 150 -s 30 55  87.1

# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 100 -s 40 80
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 150 -i 100 -s 60 120
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 55 90  
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 80 -i 200 -s 30 55 --p 0.34
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 80 -i 200 -s 30 55 --p 0.36
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 80 -i 200 -s 30 55 --p 0.38
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 80 -i 200 -s 30 55  --p 0.42
# 87 94.7 97.8 98.8
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 -e 80 -d msmt17 -i 200 -s 30 55
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 60 --tau 1.5
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 60 --tau 1.8
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 60 --tau 2
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 60 --tau 2.3
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -i 200 -s 30 60 --tau 2.5
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -d msmt17 -i 200 -s 30 60
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -d msmt17 -i 200 -s 30 60
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 -e 100 -d msmt17 -i 400 -s 30 60 80
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 --cams-decay 0.8 -e 80 -d msmt17
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 6 --gamma 2 --cams-decay 0.7 -e 60 -d msmt17
#CUDA_VISIBLE_DEVICES=0,1 python main.py --k1 30 --k2 6 --gamma 2 --cams-decay 1 -d msmt17
#CUDA_VISIBLE_DEVICES=1,2 python main.py --k1 30 --k2 6 --gamma 2 --cams-decay 0.6 -d personx
#CUDA_VISIBLE_DEVICES=1,2 python main.py --k1 30 --k2 6 --gamma 2 --cams-decay 0.6 -d veri
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 30 --k2 12 --gamma 2 --cams-decay 1
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 40 --k2 6 --gamma 2 --cams-decay 1
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --k1 40 --k2 10 --gamma 2 --cams-decay 0.9
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 60 -i 200 --global-momentum 0.2
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 60 -i 200 --global-momentum 0.3
#CUDA_VISIBLE_DEVICES=1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 80 -i 200 --global-momentum 0.08
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d msmt17 --data-dir /media/lab225/diskA/dataset/ReID-data -e 60 -i 200 --global-momentum 0.05
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d personx --data-dir /media/lab225/diskA/dataset/ReID-data -e 60 -i 200 --global-momentum 0.05
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d veri --data-dir /media/lab225/diskA/dataset/ReID-data -e 60 -i 200 --global-momentum 0.05
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 50 -i 200 --global-momentum 0.01
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 50 -i 200 --global-momentum 0.08
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data -e 50 -i 200 --global-momentum 0.1

#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.005
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.003
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.05







#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.2
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 0.5
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 1.4
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 1.5
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 1.6
#
#
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 8
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --global-momentum 9


#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --gated 5
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py -b 256 -d market1501 --data-dir /media/lab225/diskA/dataset/ReID-data --epochs 50 --gated 6


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --height 224 --width 224
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --height 224 --width 224
