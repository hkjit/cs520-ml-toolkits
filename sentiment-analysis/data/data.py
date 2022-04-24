import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('crowdflower/twitter-airline-sentiment', path="./data", unzip=True)