python -u exp/train.py \
  --dataset DF18 \
  --model TMWF \
  --gpu 6 \
  --feature DIR \
  --seq_len 30720 \
  --train_epochs 30 \
  --batch_size 80 \
  --learning_rate 5e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score P@min \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/test.py \
  --dataset DF18 \
  --model TMWF \
  --gpu 6 \
  --feature DIR \
  --seq_len 30720 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score P@min \
  --save_name max_f1