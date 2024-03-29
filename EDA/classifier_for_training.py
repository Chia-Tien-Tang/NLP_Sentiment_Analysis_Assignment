from typing import List

import torch

import pandas as pd
import numpy as np

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                          BertTokenizer, BertForSequenceClassification,
                          RobertaTokenizer, RobertaForSequenceClassification,
                          ElectraTokenizer, ElectraForSequenceClassification,
                          DebertaTokenizer, DebertaForSequenceClassification,
                          get_linear_schedule_with_warmup)

class AspectTermDataset(Dataset):
    def __init__(self, sentences, targets, tokenizer, max_len):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below
    
    def __init__(self, model_name='distilbert-base-cased'):
        """
        This should create and initilize the model. Does not take any arguments.
        
        """
        if model_name == 'distilbert-base-cased':
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        elif model_name == 'bert-base-cased':
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        elif model_name == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
        elif model_name == 'google/electra-small-discriminator':
            self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
            self.model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=3)
        elif model_name == 'microsoft/deberta-base':
            self.tokenizer = DebertaTokenizer.from_pretrained(model_name)
            self.model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
        else:
            raise ValueError("Unsupported model: " + model_name)
        self.max_len = 128

    def prepare_data_loader(self, filename: str, tokenizer, max_len, batch_size):
        df = pd.read_csv(filename, sep='	', header=None, names=['polarity', 'aspect_category', 'target_term', 'char_offsets', 'sentence'])

        # Convert polarities to numeric labels
        labels = df['polarity'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values
        sentences = df['sentence'].values
        
        dataset = AspectTermDataset(sentences=sentences, targets=labels, tokenizer=tokenizer, max_len=max_len)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        self.model = self.model.to(device)

        train_loader = self.prepare_data_loader(train_filename, self.tokenizer, self.max_len, batch_size=16)
        
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        total_steps = len(train_loader) * 3  # For 3 epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        self.model.train()

        for epoch in range(3):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)

                self.model.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)

                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
        
    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        data_loader = self.prepare_data_loader(data_filename, self.tokenizer, self.max_len, batch_size=32)

        self.model = self.model.to(device)
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().numpy())

        # Convert numeric predictions back to labels
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return [label_map[pred] for pred in predictions]