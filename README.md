# cs520-ml-toolkits
cs520 project on ML toolkits

In this project, we use two different tasks: Image Classification and Sentiment Analysis and compare/contrast wandb and comet
and their usability in Machine Learning pipelines.

The two tasks are 
1. Image Classification on MNIST
   1. The dataset for this task is gathered from torchvision package.
   2. The packages required to run this task can be found at ``requirements.txt`` and can be installed via ``pip install -r requirements.txt``
   3. To train the image classification task, run:
   ````
   python main.py
   --optimizer
   --batch_size 8
   --epochs 10
   --lr 0.001
   --use_wandb
   ````
   4. To log the experiment tracking in comet use the flag ``--use_comet`` when running the above command.
   5. Detailed instructions on different configutations can be found in README.md in ``image_classification`` folder.



2. Sentiment Analysis on Tweets
   1. The tweets are collected from Kaggle. The data can be found under this competition. https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment.
   2. Detailed instructions on data downloading and analysis can be found under README.md in ``sentiment-analysis`` folder.
   3. To train sentiment analysis on this data and log the experiment results on to Comet or Wandb, run the following commands in order:
   ````
   cd sentiment-analysis # move to the folder
   
    python main.py
    --batch_size 50
    --epochs 10
    --lr 0.001
    --train
    --use_wandb
    ````
   3. To log the experiment tracking in comet use the flag ``--use_comet`` when running the above command.
