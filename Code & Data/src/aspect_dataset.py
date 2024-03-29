import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

"""
Custom PyTorch dataset:
 - Handle text preprocessing 
 - Formatting for model training 
 - Tokenization and Encoding inputs for sentiment extraction tasks
"""

class AspectTermDataset(Dataset):
    def __init__(self, sentences, terms, offsets, aspect_categories, labels, tokenizer, max_len):
        self.sentences = sentences
        self.terms = terms
        self.offsets = offsets
        self.aspect_categories = aspect_categories
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        term = str(self.terms[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }