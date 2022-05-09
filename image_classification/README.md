1. Set your current working directory to image_classification
2. To train the model and use wandb for experiment tracking, run the following command
````
python main.py --optimizer --batch_size 8 --epochs 10
--lr 0.001 --use_wandb
````
3. To train the model and use comet for experiment tracking, run the following command
````
python main.py --optimizer --batch_size 8 --epochs 10
--lr 0.001 --use_comet
````
4. To run hyperparmeter sweeps in wandb use ``--sweep`` argument.