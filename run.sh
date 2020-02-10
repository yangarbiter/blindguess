

python ./main.py --experiment experiment01 \
  --no-hooks \
  --norm inf --eps 0.1 \
  --dataset fashion \
  --model ce-tor2-CNN002 \
  --attack pgd \
  --optimizer sgd \
  --learning_rate 0.001 \
  --epochs 60 \
  --momentum 0.9 \
  --weight_decay 0 \
  --batch_size 128 \
  --random_seed 0
