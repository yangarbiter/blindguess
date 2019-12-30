#python ./main.py --experiment experiment01 \
#  --no-hooks \
#  --norm inf --eps 0.1 \
#  --dataset mnist \
#  --model advce-tor-CNN002 \
#  --attack pgd \
#  --random_seed 0

python ./main.py --experiment experiment02 \
  --no-hooks \
  --norm inf --eps 0.1 \
  --dataset mnist \
  --model ce-tor-CNN001 \
  --attack multitarget \
  --random_seed 0

#python ./main.py --experiment experiment01 \
#  --no-hooks \
#  --norm inf --eps 0.031 \
#  --dataset svhn \
#  --model ce-tor-WRN_40_10 \
#  --attack pgd \
#  --random_seed 0
