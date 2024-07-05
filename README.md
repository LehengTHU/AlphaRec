<div align=center>

<h1>Language Models Encode Collaborative Signals in Recommendation</h1>

<img src="https://img.shields.io/badge/License-MIT-blue" alt="license">

<div>
      <a href="https://lehengthu.github.io//" target="_blank">Leheng Sheng</a><sup>1</sup>,
      <a href="https://anzhang314.github.io/" target="_blank">An Zhang</a><sup>1</sup><sup>*</sup>,
    <a href="https://github.com/zy20031230/" target="_blank">Yi Zhang</a><sup>2</sup>,
    <a href="https://github.com/chenyuxin1999/" target="_blank">Yuxin Chen</a><sup>1</sup>,
      <a href="https://xiangwang1223.github.io./" target="_blank">Xiang Wang</a><sup>2</sup>,
      <a href="https://www.chuatatseng.com/" target="_blank">Tat-Seng Chua</a><sup>1</sup>,

<div>
  <sup>1</sup>National University of Singapore, <sup>2</sup>University of Science and Technology of China
       </div>   
<div>
<sup>*</sup>Corresponding author. 
   </div>

</div>

<h2>Homomorphism</h2> 

<div style="width: 100%; text-align: center; margin:auto;">
      <!-- <img style="width: 100%" src="assets/intro-tSNE.png"> -->
    <img style="width: 100%" src="assets/main_figure.png">
</div>

<h2>Rapid Convergence</h2> 

<div style="width: 55%; text-align: center; margin:auto;">
      <img style="width: 55%" src="assets/convergence.png">
</div>

<h2>User Intention Capture</h2> 

<div style="width: 80%; text-align: center; margin:auto;">
      <img style="width: 80%" src="assets/case-intention.png">
</div>
<!-- ![world](assets/intro-tSNE.png)
![case](assets/case-intention.png) -->


</div>


## 📋 TODO 

- [ ] Release user intention capture datasets.
- [ ] ...

## 👉 Quick Start for AlphaRec

### Dependencies

Our experiments have been tested on Python 3.9.12 with PyTorch 1.13.1+cu117. 🛎️ Python version over 3.10 may lead to some bugs in the package 'reckit'.

1. To install required packages, using the following command:
```bash
pip install -r requirements.txt
```

2. Before using the general recommendation, run the following command to install the evaluator:
```bash
pushd models/General/base
python setup.py build_ext --inplace
popd
```

### Dataset downloading

Please download the datasets from the following anonymous link and put the unzipped dataset in the `data` folder:

https://drive.google.com/drive/folders/1iGKeTx3vqCtbeVdWkHOwgpbY3-s7QDy_?usp=sharing

Example of the file structure:
```
├── assets/
├── models/
├── data/
    ├── General/
        ├── amazon_movie/ # target datasets
            ├── cf_data/
            ├── item_info/
```

### Commands for running 
Books
```bash
nohup python main.py --rs_type General --clear_checkpoints --saveID tau_0.15_v3_mlp_ --dataset amazon_book_2014 --model_name AlphaRec --n_layers 2 --patience 20 --cuda 0 --no_wandb --train_norm --pred_norm --neg_sample 256 --lm_model v3 --model_version mlp --tau 0.15 --infonce 1 &>logs/amazon_book_2014_tau_0.15_v3_mlp__2.log &
```
Movies & TV
```bash
nohup python main.py --rs_type General --clear_checkpoints --saveID tau_0.15_v3_mlp_ --dataset amazon_movie --model_name AlphaRec --n_layers 2 --patience 20 --cuda 1 --no_wandb --train_norm --pred_norm --neg_sample 256 --lm_model v3 --model_version mlp --tau 0.15 --infonce 1 &>logs/amazon_movie_tau_0.15_v3_mlp__2.log &
```

Games
```bash
nohup python main.py --rs_type General --clear_checkpoints --saveID tau_0.2_v3_mlp_ --dataset amazon_game --model_name AlphaRec --n_layers 2 --patience 20 --cuda 2 --no_wandb --train_norm --pred_norm --neg_sample 256 --lm_model v3 --model_version mlp --tau 0.2 --infonce 1 &>logs/amazon_game_tau_0.2_v3_mlp__2.log &
```

## ☎️ Contact

Please contact the first author of this paper for queries.

- Leheng Sheng, leheng.sheng@u.nus.edu