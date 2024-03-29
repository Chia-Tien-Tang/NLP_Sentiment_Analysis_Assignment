from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Transformers
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

# Custom .py
from utils import polarity_to_numerical, map_prediction_to_label
from aspect_dataset import AspectTermDataset
from custom_layers import MeanPooling

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below
    
    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 128
        
        # 3 classes (positive, negative, neutral)
        # The output size of BertModel is 768 for 'bert-base-uncased'
        self.classifier = nn.Linear(768, 3).to(self.device)

        # Initialization of the custom mean pooling layer
        self.mean_pooling_layer = MeanPooling().to(self.device)
    
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        # Load training and development data from CSV files
        # rename columns' names for [polarity, aspect category, target term, character offsets, sentence]
        train_df = pd.read_csv(train_filename, sep='	', header=None, names=['polarity', 'aspect_category', 'target_term', 'char_offsets', 'sentence'])
        dev_df = pd.read_csv(dev_filename, sep='	', header=None, names=['polarity', 'aspect_category', 'target_term', 'char_offsets', 'sentence'])
        
        # Convert polarity labels to numerical format for both training and development sets
        train_labels = train_df['polarity'].apply(polarity_to_numerical).values
        dev_labels = dev_df['polarity'].apply(polarity_to_numerical).values
        
        # Create custom datasets for training and development sets
        train_dataset = AspectTermDataset(
            sentences=train_df['sentence'].values,
            terms=train_df['target_term'].values,
            offsets=train_df['char_offsets'].values,
            aspect_categories=train_df['aspect_category'].values,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

        dev_dataset = AspectTermDataset(
            sentences=dev_df['sentence'].values,
            terms=dev_df['target_term'].values,
            offsets=dev_df['char_offsets'].values,
            aspect_categories=dev_df['aspect_category'].values,
            labels=dev_labels,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        
        # Prepare DataLoader for batching operations
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
        
        # Training setup: epochs, model preparation, optimizer, and scheduler
        epochs = 3 # Adjust based on the need
        self.model = self.model.to(device)
        mean_pooling_layer = MeanPooling().to(device)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            n_examples = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                mean_pooled = mean_pooling_layer(outputs, attention_mask)
                logits = self.classifier(mean_pooled)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Calculate the accuracy
                _, predicted_labels = torch.max(logits, dim=1)
                correct_predictions += torch.sum(predicted_labels == labels).item()
                n_examples += labels.size(0)
            
            # Print training progress
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = correct_predictions / n_examples * 100
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            
        # (Optional) Save the model into .pt file
        torch.save(self.model.state_dict(), "bert_sentiment_model.pt")

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        # Load dataset
        test_df = pd.read_csv(data_filename, sep='	', header=None, names=['polarity', 'aspect_category', 'target_term', 'char_offsets', 'sentence'])
        
        # Create custom dataset for the test set (labels are placeholders as they're not used for prediction)
        test_dataset = AspectTermDataset(
            sentences=test_df['sentence'].values,
            terms=test_df['target_term'].values,
            offsets=test_df['char_offsets'].values,
            aspect_categories=test_df['aspect_category'].values,
            labels=np.zeros((len(test_df),)),  # Placeholder labels as they're not used in prediction
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Ensure the model is on the correct device and set it to evaluation mode
        self.model = self.model.to(device)

        self.model.eval()
        predictions = []
                
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                mean_pooled = self.mean_pooling_layer(outputs, attention_mask)
                logits = self.classifier(mean_pooled)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        
        # Convert numerical predictions back to label strings
        predictions_labels = [map_prediction_to_label(pred) for pred in predictions]
        
        return predictions_labels