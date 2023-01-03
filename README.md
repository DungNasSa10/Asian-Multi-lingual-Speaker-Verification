# **Asian-Multi-lingual-Speaker-Verification**


This is our solution (No_Train_No_Gain team) for task T01 and T02 in O-COCOSDA and VLSP 2022 - A-MSV Shared task: Asian Multilingual Speaker Verification (Top 2 Private test in both tasks).

## **Installation**

- Python version == 3.8

``` 
conda create -n amsv python=3.8 -y
conda activate amsv
pip install -r requirements.txt
```

## **Data and Output Folder**

```
data
|
└───MSV_CommonVoice_data
|   |
│   │───metadata
|   |   all_new_metadata.txt
|   |   all_new_metadata2.txt
|   |   train_list_vi.txt
|   |   ...
|   |    
│   │───unzip_data
|   |   |
|   |   |───vi
|   |   |    |───clips
|   |   |
|   |   ...
│   │
│   |───aug2
|   |   |───vi
|   |   |    |───clips
│   |   │   
│   |   ...
|   
|───MSV_Youtube_data
|
|───musan_augment
|
|───rir_aug

output
|
|───embedding_data
|   
|───submission
|
|───ecapa
|
...
```

## **Training Process**

### **Offline data augmentation**

- For each speaker having less than 6 utterances, we used RIRs and MUSAN to create extra 5 copies of his/her randomly selected utterances with the following script:
```
python src/aug.py
```

### **Traning models with MSV_CommonVoice_data**
- ECAPA_TDNN with CosineMargin loss (the model checkpoints will be stored in folder ```output/ecapa/model```)
```
python src/trainSpeakerNet.py --config src/configs/ECAPA_TDNN_CM.yaml
```
- CNN_TDNN with AAMSoftmax loss
```
python src/trainSpeakerNet.py --config src/configs/CNN_TDNN_AAM.yaml
```

You can change the arguments in configuration file or pass individual arguments that are defined in trainSpeakerNet.py by --{ARG_NAME} {VALUE}. Note that the configuration file overrides the arguments passed via command line.

### **Clustering MSV_Youtube_data**
- Clustering for each video for Vietnam 
```
python src/clustering.py --config src/configs/ECAPA_TDNN_yt.yaml
```
- Clustering for all video for Vietnam
```
python src/clustering.py --config src/configs/ECAPA_TDNN_yt.yaml --all
```
You need to change language in configuration file to cluster for each language

### **Traning with MSV_Youtube_data**
- We use result of clustering for each video to create new training data and fine-tune model with this data set
- This script is used to warm up the output FC layer (5-10 epochs)
```
python src/trainSpeakerNet.py --config src/configs/CNN_TDNN_AAM.yaml --freeze --initial_model output/ecapa/model/model000000034.model
```
- Then, we fine-tune entire network with small learning rate:
```
python src/trainSpeakerNet.py --config src/configs/CNN_TDNN_AAM.yaml --initial_model model_checkpoint --lr 0.0001
```
## Inference

These script will output submission file for Vietnamese private test in ```output/submission/t1/private_test```
### Cosine similarity score
```
python src/trainSpeakerNet.py --config src/configs/CNN_TDNN_CM.yaml --initial_model model_checkpoint --test
```

### AS-norm (The best result in public test)
```
python src/asnorm.py --config src/configs/asnorm.yaml --initial_model model_checkpoint
```

### PLDA

```
python plda_score.py --config src/configs/CNN_TDNN_AAM.yaml --initial_model model_checkpoint
```
