# TFCE & TFCE-MLP

This repository contains the implementation of **TFCE** and **TFCE-MLP**, developed for recommendation systems.  
It is a fork of the original [AlphaRec](https://github.com/LehengTHU/AlphaRec) repository.

---

## 🛠 Installation

Our experiments were tested on **Python 3.9.12** with **PyTorch 1.13.1+cu117**.  
⚠️ Python versions above 3.10 may cause issues with the `reckit` package.

### Step 1: Environment Setup

Set up a virtual environment and install PyTorch manually (as per your CUDA version) from the [official site](https://pytorch.org/get-started/previous-versions/).  
Then install the remaining dependencies:

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
TFCE-MLP
```bash
# Amazon Game
nohup python main.py --rs_type General --clear_checkpoints --saveID tfcemlp --dataset amazon_game --model_name TFCEMLP --n_layers 3 --patience 20 --cuda 0 --no_wandb --train_norm --pred_norm --neg_sample 512 --lm_model v3 --model_version mlp --tau 0.2 --infonce 1 --verbose 1 --no-is_one_pos_item --n_pos_samples 3 --hidden_size 128 &> logs/amazon_game.log &

# Amazon Movie
nohup python main.py --rs_type General --clear_checkpoints --saveID tfcemlp --dataset amazon_movie --model_name TFCEMLP --n_layers 2 --patience 20 --cuda 0 --no_wandb --train_norm --pred_norm --neg_sample 512 --lm_model v3 --model_version mlp --tau 0.15 --infonce 1 --verbose 1 --no-is_one_pos_item --n_pos_samples 9 --hidden_size 128 &> logs/amazon_movie.log &

# Amazon Book
nohup python main.py --rs_type General --clear_checkpoints --saveID tfcemlp --dataset amazon_book --model_name TFCEMLP --n_layers 3 --patience 20 --cuda 0 --no_wandb --train_norm --pred_norm --neg_sample 512 --lm_model v3 --model_version mlp --tau 0.15 --infonce 1 --verbose 1 --no-is_one_pos_item --n_pos_samples 7  --hidden_size 256 &> logs/amazon_book.log &```
```

TFCE
```bash
# Amazon Game
nohup python main.py --rs_type General --clear_checkpoints --saveID tfce --dataset amazon_game --model_name TFCE --n_layers 4 --cuda 0 --no_wandb --train_norm --pred_norm --lm_model v3 --model_version mlp --infonce 1 --verbose 1 &> logs/amazon_game.log &

# Amazon Movie
nohup python main.py --rs_type General --clear_checkpoints --saveID tfce --dataset amazon_movie --model_name TFCE --n_layers 3  --cuda 0 --no_wandb --train_norm --pred_norm  --lm_model v3 --model_version mlp --infonce 1 --verbose 1 &> logs/amazon_movie.log &

# Amazon Book
nohup python main.py --rs_type General --clear_checkpoints --saveID tfce --dataset amazon_book --model_name TFCE --n_layers 5 --cuda 0 --no_wandb --train_norm --pred_norm  --lm_model v3 --model_version mlp --infonce 1 --verbose 1 &> logs/amazon_book.log &
```

## ☎️ Contact

Please contact the first author of this paper for queries.

- Leheng Sheng, leheng.sheng@u.nus.edu

## 🌟 Citation

You can cite this paper as follows if you find our work helpful:

```bibtex
@article{AlphaRec,
  title={Language Models Encode Collaborative Signals in Recommendation},
  author={Sheng, Leheng and Zhang, An and Zhang, Yi and Chen, Yuxin and Wang, Xiang and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2407.05441},
  year={2024}
}
```
