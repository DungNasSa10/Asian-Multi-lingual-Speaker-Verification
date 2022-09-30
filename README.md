# Asian-Multi-lingual-Speaker-Verification

## Installation, training and evaluation

### Installation

- Python version == 3.8

``` 
    conda create -n amsv python=3.8 -y
    conda activate amsv
    pip install -r requirements.txt
```

### Data Folder

- data
    - MSV_CommonVoice_data
        - metadata
    - MSV_Youtube_data

> **Note:** 
- /temporary/ : This folder contains all temporary files and will be deleted when releasing. 

### Traning

> python src/ECAPA-TDNN/trainECAPAModel.py --max_epoch 50 --batch_size 16 --train_list data/MSV_CommonVoice_data/metadata/all_new_metadata.txt --train_path /home2/vietvq/UNDERFITT/data/MSV_CommonVoice_data/unzip_data/ --eval_list data/Test/veri_test2.txt --eval_path data/Test/wav/ --musan_path data/musan_augment/ --rir_path data/rirs_noises/ --save_path output/classification_common/ --n_class 17714
