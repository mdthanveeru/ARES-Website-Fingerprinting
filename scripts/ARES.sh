for num_tabs in 2 3 4 5
do
  for scenario in Closed Open
    do
      dataset=${scenario}_${num_tabs}tab

      for filename in train valid test
      do 
          python -u exp/dataset_process/gen_mtaf.py \
            --dataset ${dataset} \
            --seq_len 10000 \
            --in_file ${filename}
      done

      python -u exp/train.py \
        --dataset closed_2tab \
        --model ARES \
        --device cuda:0 \
        --num_tabs 2 \
        --train_file mtaf_train \
        --valid_file mtaf_valid \
        --feature MTAF \
        --seq_len 8000 \
        --train_epochs 300 \
        --batch_size 256 \
        --learning_rate 1e-3 \
        --optimizer AdamW \
        --loss MultiLabelSoftMarginLoss \
        --eval_metrics AUC P@2 AP@2 \
        --save_metric AP@2 \
        --save_name base

      python -u exp/test.py \
        --dataset closed_2tab \
        --model ARES \
        --device cuda:0 \
        --num_tabs 2 \
        --valid_file mtaf_valid \
        --test_file mtaf_test  \
        --feature MTAF \
        --seq_len 8000 \
        --batch_size 256 \
        --eval_metrics AUC P@2AP@2 \
        --load_name base
    done
done