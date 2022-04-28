1. Install kaggle package to download the dataset using 
``pip3 install kaggle``
   
2. ``Go to your Kaggle account Tab at https://www.kaggle.com/<username>/account and click ‘Create API Token’. A file named kaggle.json will be downloaded. Move this file in to ~/.kaggle/ folder in Mac and Linux or to C:\Users<username>.kaggle\ on windows.``

3. Set your current working directory to ``sentiment-analysis``

4. Run ``python data/data.py``. this will download dataset in ``data`` directory with files ``database.sqlite`` and ``Tweets.csv``

5. Perform data-analysis and preprocessing in `data-analysis` folder. Further details about data analysis are provided 
in Readme file in `data-analysis`.
   
6. After data analysis is completed we generate a `preprocess_tweets.csv` on which we do training and testing.
