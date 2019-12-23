python ./main.py --experiment experiment01 \
  --no-hooks \
  --norm 2 --eps 1.0 \
  --dataset smallcifar10 \
  --model ce-tor-WideResNet \
  --attack pgd \
  --random_seed 0