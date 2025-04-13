import os
import random
import wandb
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import BertTokenizer, WordpieceTokenizer
from transformers.tokenization_utils import Trie

from dataset import SMILESDataset_Finetune
from calc_property import calculate_property
from SPMM_models import SPMM, SPMMDownstreamPropertyPredictor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class ConfigDict(dict):
    def __getattr__(self, name):
        if name in self:
            value = self[name]

            if isinstance(value, dict):
                return ConfigDict(value)
            
            return value
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class Trainer:
    def __init__(self, config):
        self.config = ConfigDict(config)

        set_seed(self.config.seed)

        self.model = SPMMDownstreamPropertyPredictor(self.config, property_width=768)
        self.load_pretrained_modules()

        # Freeze modules
        for name, param in self.model.named_parameters():
            # if any(name.startswith(mod) for mod in ['text_encoder', 'property_encoder', 'property_embed', 'property_proj']):
            if any(name.startswith(mod) for mod in ['text_encoder', 'property_encoder', 'property_embed']):
                param.requires_grad = False

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs, eta_min=1e-7)

        # Get data
        train_data, val_data, test_data = self.split_csv_dataset()

        self.mean = None # rain_data['PDSC'].mean()
        self.std = None # train_data['PDSC'].std()

        train_dataset = SMILESDataset_Finetune(data=train_data, data_length=None, shuffle=True, mean=self.mean, std=self.std)
        val_dataset = SMILESDataset_Finetune(data=val_data, data_length=None, shuffle=False, mean=self.mean, std=self.std)
        test_dataset = SMILESDataset_Finetune(data=test_data, data_length=None, shuffle=False, mean=self.mean, std=self.std)

        print(f'Train Size: {train_dataset.__len__()}, Val Size: {val_dataset.__len__()}, Test Size: {test_dataset.__len__()}')

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Get tokenizer
        self.tokenizer = BertTokenizer(vocab_file=self.config.vocab_path, do_lower_case=False, do_basic_tokenize=False)
        self.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.tokenizer.vocab, unk_token=self.tokenizer.unk_token, max_input_chars_per_word=250)

        # Get loss function
        self.loss_function = F.l1_loss

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def split_csv_dataset(self):
        val_size = self.config.val_size
        test_size = self.config.test_size

        # Load dataset
        df = pd.read_csv(self.config.data_path)

        # Check that val + test < 1.0
        assert 0 < val_size < 1 and 0 < test_size < 1, "val_size and test_size must be in (0, 1)"
        assert val_size + test_size < 1, "val_size + test_size must be less than 1"

        # Remaining for training
        train_size = 1.0 - val_size - test_size
        train_df, temp_df = train_test_split(
            df,
            test_size=val_size + test_size,
            shuffle=True,
            random_state=self.config.seed,
        )

        val_prop = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_prop,  
            shuffle=True,
            random_state=self.config.seed,
        )

        return train_df, val_df, test_df

    def load_pretrained_modules(self):
        ''' Load pretrained weights for text / property encoders '''

        with torch.serialization.safe_globals([BertTokenizer, Trie, WordpieceTokenizer]):
            state_dict = torch.load(self.config.pretrained_weights_path, map_location='cpu')['state_dict']

            text_encoder_state_dict = {k.replace("text_encoder.", ""): v for k, v in state_dict.items() if k.startswith("text_encoder.")}
            property_encoder_state_dict = {k.replace("property_encoder.", ""): v for k, v in state_dict.items() if k.startswith("property_encoder.")}
            property_embed_state_dict = {k.replace("property_embed.", ""): v for k, v in state_dict.items() if k.startswith("property_embed.")}
            # property_proj_state_dict = {k.replace("property_proj.", ""): v for k, v in state_dict.items() if k.startswith("property_proj.")}
        
            self.model.text_encoder.load_state_dict(text_encoder_state_dict)
            self.model.property_encoder.load_state_dict(property_encoder_state_dict)
            self.model.property_embed.load_state_dict(property_embed_state_dict)
            # self.model.property_proj.load_state_dict(property_proj_state_dict)

            self.model.text_encoder.cls = nn.Identity()
            self.model.property_encoder.cls = nn.Identity()

    def run(self):
        best_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), '../pdsc_predictor.ckpt')

            if self.mean is not None and self.std is not None:
                train_loss = train_loss * self.std + self.mean
                val_loss = val_loss * self.std + self.mean

            print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
            # wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        test_loss = self.test()
        if self.mean is not None and self.std is not None:
            test_loss = test_loss * self.std + self.mean

        print(f'Test Loss: {test_loss:.5f}')
        # wandb.log({"test_loss": test_loss})

    def train_epoch(self):
        self.model.train()

        total_loss = 0.0
        for step, batch_dict in enumerate(tqdm(self.train_dataloader)):
            self.optimizer.zero_grad()

            pv, smiles, target = batch_dict['properties'], batch_dict['smiles'], batch_dict['target']
            pv, target = pv.to(self.device), target.to(self.device)

            text_input = self.tokenizer(smiles, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(self.device)

            out = self.model(property_original=pv, text_input_ids=text_input.input_ids[:, 1:])
            loss = self.loss_function(out.squeeze(1), target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        self.scheduler.step()
        
        return total_loss / len(self.train_dataloader)
    
    def val_epoch(self):
        self.model.eval()

        total_loss = 0.0
        for step, batch_dict in enumerate(tqdm(self.val_dataloader)):
            pv, smiles, target = batch_dict['properties'], batch_dict['smiles'], batch_dict['target']
            pv, target = pv.to(self.device), target.to(self.device)

            text_input = self.tokenizer(smiles, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model(property_original=pv, text_input_ids=text_input.input_ids[:, 1:])

            loss = self.loss_function(out.squeeze(1), target)
            total_loss += loss.item()

        return total_loss / len(self.val_dataloader)
    
    def test(self):
        self.model.eval()

        total_loss = 0.0
        for step, batch_dict in enumerate(tqdm(self.test_dataloader)):
            pv, smiles, target = batch_dict['properties'], batch_dict['smiles'], batch_dict['target']
            pv, target = pv.to(self.device), target.to(self.device)

            text_input = self.tokenizer(smiles, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model(property_original=pv, text_input_ids=text_input.input_ids[:, 1:])

            loss = self.loss_function(out.squeeze(1), target)
            total_loss += loss.item()

        return total_loss / len(self.test_dataloader)
    

if __name__ == '__main__':
    finetune_config = {
        'seed': 42,

        'data_path': '../data/data_new_spmm.csv',
        'pretrained_weights_path': '../SPMM_Checkpoint.ckpt',
        'vocab_path': './vocab_bpe_300.txt',

        'val_size': 0.1,
        'test_size': 0.1,

        'num_epochs': 30,
        'lr': 0.01,

        'embed_dim': 256, # 256,
        'batch_size': 32,
        'regression_head_hidden_dim': 64,

        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }

    trainer = Trainer(finetune_config)
    trainer.run()




    



