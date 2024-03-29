import torch
import torch.nn as nn


"""
Summary of MeanPooling:
The module processes the last hidden state from a pre-trained LM (In our case, we use BERT), applying an attention mask to focus on relevant tokens. 

It computes the mean pooled representation by considering only the masked (non-padded) tokens, enhancing the model's capability to understand aspect-specific contexts. This approach is inspired by advances in ABSA that leverage aspect-specific input transformations and fine-tuned pre-trained language models to better capture the nuances of sentiment related to specific aspects in sentences. 

Reference Paper: "Aspect-specific Context Modeling for Aspect-based Sentiment Analysis" (Ma et al., arXiv:2207.08099).
"""

class MeanPooling(nn.Module):
    def forward(self, outputs, attention_mask):
        # outputs is a tuple returned by BertModel, containing (last_hidden_state, pooler_output, hidden_states, attentions)
        # Extract the token embeddings from the first element of the outputs tuple
        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled