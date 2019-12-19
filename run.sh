python ./main.py --experiment experiment01 \
  --no-hooks \
  --norm 2 --eps 1.0 \
  --dataset smallmnist \
  --model ce-tor-CNN001 \
  --attack pgd \
  --random_seed 0