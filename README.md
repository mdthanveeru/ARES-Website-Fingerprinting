# ARES - Website Fingerprinting
Includes code of ARES model for website fingerprinting attack

## WFlib Overview

## Usage

### Install 

```sh
git clone https://github.com/mdthanveeru/ARES-Website-Fingerprinting.git
cd ARES-Website-Fingerprinting
```


**Create a conda environment**

- install miniconda/anaconda in your device

```sh
conda create --name ares python=3.8
conda activate ares
pip install -e .
```
You can pip install if any package is missing in the environment.

### Datasets

```sh
mkdir datasets
```

- Download datasets ([link](https://zenodo.org/records/13732130)) and place it in the folder `./datasets`


- The extracted dataset is in npz format and contains two values: X and y. X represents the cell sequence, with values being the direction (e.g., 1 or -1) multiplied by the timestamp. y corresponds to the labels. Note that the input of some datasets consists only of direction sequences.

- Divide the dataset into training, validation, and test sets.

```sh
# For multi-tab datasets
python exp/dataset_process/dataset_split.py --dataset closed_2tab --use_stratify False
```

### Training \& Evaluation

1. Make feature datasets
```sh
dataset=closed_2tab # change the dataset name accordingly(closed_3tab,open_3tab...etc)

      for filename in train valid test
      do 
          python -u exp/dataset_process/gen_mtaf.py \
            --dataset ${dataset} \
            --seq_len 10000 \
            --in_file ${filename}
      done
```
2. Train the model
```sh
scenario="closed/open"  #select the scenario of dataset open or closed
num_tabs="2/3/4/5" #select the number of tabs

dataset=${scenario}_${num_tabs}tab

python -u exp/train.py \
  --dataset ${dataset} \
  --model ARES \
  --device cuda:0 \
  --num_tabs ${num_tabs} \
  --train_file mtaf_train \
  --valid_file mtaf_valid \
  --feature MTAF \
  --seq_len 8000 \
  --train_epochs 300 \
  --batch_size 256 \
  --learning_rate 1e-3 \
  --optimizer AdamW \
  --loss MultiLabelSoftMarginLoss \
  --eval_metrics AUC P@${num_tabs} AP@${num_tabs} \
  --save_metric AP@${num_tabs} \
  --save_name base

```

2. Test the model
```sh
scenario="closed/open"  #select the scenario of dataset open or closed
num_tabs="2/3/4/5" #select the number of tabs

dataset=${scenario}_${num_tabs}tab

python -u exp/test.py \
        --dataset ${dataset} \
        --model ARES \
        --device cuda:0 \
        --num_tabs ${num_tabs} \
        --valid_file mtaf_valid \
        --test_file mtaf_test  \
        --feature MTAF \
        --seq_len 8000 \
        --batch_size 256 \
        --eval_metrics AUC P@${num_tabs} AP@${num_tabs} \
        --load_name base

```




## Contact
If you have any questions or suggestions, feel free to contact:

- Muhammed Thanveer U (thanveeru123@gmail.com)
