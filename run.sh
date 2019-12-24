python ./main.py --experiment experiment01 \
  --no-hooks \
  --norm inf --eps 0.031 \
  --dataset mnist \
  --model ce-tor-CNN001 \
  --attack pgd \
  --random_seed 0
