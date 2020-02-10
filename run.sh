

python ./main.py --experiment experiment01 \
  --no-hooks \
  --norm inf --eps 0.1 \
  --dataset fashion \
  --model trades6ce-tor-WRN_40_10 \
  --attack pgd \
  --optimizer sgd \
  --learning_rate 0.0001 \
  --epochs 60 \
  --momentum 0.9 \
  --weight_decay 0 \
  --random_seed 0
