# 2022AICUP
Competition URL: https://aidea-web.tw/topic/5f632f38-7213-4d4d-bea3-117ff13c1afb \
Private Leaderboard: 28 / 153 (Top 25%)

# Getting started
- Clone this repo to your local
``` bash
git clone https://github.com/ziheshen/2022AICUP.git
cd 2022AICUP
```

# Data preprocessing
- First, download the dataset from the [official](https://aidea-web.tw/topic/93c8c26b-0e96-44bc-9a53-1c96353ad340). And put all the data into a director, named `../dataset/private`. Then use the following command to resize the images into 384 * 384 and split the data into train/valid sets. 
``` bash
python preprocessing.py
```

# Inference
- First, download the pretrained models from [here](https://drive.google.com/drive/folders/1-Ulztgd0y8F5Uoon9ILACAM4h5p7MONI?usp=share_link). And put the models into the director `./checkpoints`. Then use the following command to do model inference (You may need to change the dataset path on your own).
``` bash
python test.py --root path/to/public_data
```
- In the public data folder, it should contain the directory `images` and the file `submission_example.csv`.

# Training
- After preparing the data by those mentioned above, you could use the script `train.sh` to train the model from scratch. Please see more detail in this script if you want to train your model.
``` bash
bash train.sh
```

# Grad-cam visualization
- After training, we used grad cam to visualize where the model focuses. The visualization results are shown below.
<img src="https://drive.google.com/file/d/1yjQma8EPTuzZO6mh35OqDBzxbs5F8XJ8/view?usp=share_link" width=41% height=41%>|<img src="https://github.com/come880412/crop_classification/blob/main/images/20180626-3-0028.jpg" width=40% height=40%>
<img src="https://github.com/come880412/crop_classification/blob/main/images/160118-3-0086.jpg" width=41% height=41%>|<img src="https://github.com/come880412/crop_classification/blob/main/images/20170205-1-0021.jpg" width=40% height=40%>

- These results show that our model learns the most important features in the corresponding class, instead of overfitting on some unimportant features.
- If you have any questions, feel free to send me an email! come880412@gmail.com
