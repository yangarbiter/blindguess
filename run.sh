
#python ./main.py --experiment restrictedImgnet2 \
#  --norm inf --eps 0.005 \
#  --dataset resImgnet112v3 \
#  --model ce-tor-ResNet152-bs128 \
#  --attack pgd \
#  --random_seed 0
#
#python ./main.py --experiment restrictedImgnet2 \
#  --norm inf --eps 0.005 \
#  --dataset resImgnet112v2 \
#  --model ce-tor-ResNet101-bs128 \
#  --attack pgd \
#  --random_seed 0

python ./main.py --experiment restrictedImgnet2 \
  --norm inf --eps 0.005 \
  --dataset resImgnet112v3 \
  --model advce-tor-ResNet152-adambs128 \
  --attack pgd \
  --random_seed 0

#python ./main.py --experiment restrictedImgnet \
#  --norm inf --eps 0.005 \
#  --dataset resImgnet112v3 \
#  --model strades6ce-tor-ResNet152-adambs128 \
#  --attack pgd \
#  --random_seed 0
#
#python ./main.py --experiment restrictedImgnet2 \
#  --norm inf --eps 0.005 \
#  --dataset resImgnet112v3 \
#  --model strades6ce-tor-ResNet152-adambs128 \
#  --attack pgd \
#  --random_seed 0

#python ./main.py --experiment experiment01 \
#  --no-hooks \
#  --norm inf --eps 0.1 \
#  --dataset mnist \
#  --model advce-torv2-CNN002 \
#  --attack pgd \
#  --random_seed 0

#python ./main.py --experiment experiment01 \
#  --no-hooks \
#  --norm inf --eps 0.031 \
#  --dataset tinyimgnet \
#  --model aug02-strades6ce-tor-tWRN50_4-bs256 \
#  --attack pgd \
#  --random_seed 0

#python ./main.py --experiment experiment01 \
#  --no-hooks \
#  --norm inf --eps 0.031 \
#  --dataset tinyimgnet \
#  --model aug01-ce-tor-tWRN50_5 \
#  --attack pgd \
#  --random_seed 0

#python ./main.py --experiment experiment01 \
#  --no-hooks \
#  --norm inf --eps 0.031 \
#  --dataset svhn \
#  --model ce-tor-WRN_40_10 \
#  --attack pgd \
#  --random_seed 0
