# NLP_Sentiment_Analysis_Assignment
## Contributors

Chia Tien Tang, Zheng Wan, Weijing Zeng, Killian Le Metayer

## Overall Structure 
**Code & Data**
- `data`: This directory contains dataset files `devdata.csv` and `traindata.csv`
**src**
- `aspect_dataset.py`: Defines a custom dataset `AspectTermDataset`, used for loading and processing aspect-based sentiment data for training.
- `classifier.py`: Contains the definition of sentiment classifier. This is where the model architecture and training/prediction methods are implemented.
- `custom_layers.py`: Inspired by the reference paper. It contains the MeanPooling class by computing the mean pooled representation of relevant tokens from the last hidden state of BERT model by using an attention mask.This is used by `classifier.py`.
- `tester.py`: Code to evaluate the classifier's performance.
- `utils.py`: Utilities file for common functions that transform polarity labels between categorical and numerical representations.

**EDA**
- `EDA.ipynb`: Used for exploratory data analysis.
- `check_version.ipynb`: Check and ensure that the correct versions of libraries are being used.
- `classifier_for_training.py`: Where different models were compared. 

**Reference Paper**
- `220708099.pdf`: "Aspect-specific Context Modeling for Aspect-based Sentiment Analysis" (Ma et al., arXiv:2207.08099).

**Others**
- `README.md`: Markdown file providing an overview of the repository, instructions on how to use the code, install dependencies, or other general information.
- `nlp_assignment_doc_2024.pdf`: Project documentation detailing specifications and requirements.
- `Results.png`: A screenshot of model results .

## Exploratory Data Analysis（EDA）

To gain a better understanding of the dataset, we conducted EDA.

The dataset consists of five columns, namely:
1. **Polarity**: Labels indicating the sentiment polarity of opinions, including positive, negative, and neutral.
2. **Aspect Category**: Categories representing the aspects targeted by the comments, such as "service" or "food quality". There are a total of 12 different aspects.
3. **Target Term**: Specific terms or entities mentioned in the comments.
4. **Character Offsets**: The position of the target term within the sentence, represented as character indices (e.g., start index:end index).
5. **Sentence**: Complete sentences containing the target terms. The sentiment polarity is expressed with respect to the specific term in these sentences.

**What we Explored**:

- Both the training and validation sets are imbalanced datasets, particularly in terms of the "neutral" polarity.
- The same sentence may appear multiple times in the dataset due to different aspects or target terms, each time potentially having a different polarity evaluation.
- If a review contains a negative sentiment for a given aspect, other aspects will most likely be rated negative as well.

After gaining a better understanding of the dataset, we proceeded to explore various types of classification models along with different input and feature representations.

## Model Input and Feature Extraction Process

1. **Input Preprocessing**: We first loaded the data and converted the polarity labels into numerical format.The Classifier then accepts all the feature columns as input.When predicting,polarity labels are then labeled back.

2. **Formatting and Tokenization**: We performed formatting and automation from text to model input. We utilized the `encode_plus` method of the BERT tokenizer to process input sentences. The tokenizer added special tokens, [CLS] and [SEP], at the beginning and end of each sentence. The maximum sequence length output by the tokenizer was set to 128. For sentences shorter than the maximum sequence length, padding tokens were added. If the length of a sentence exceeded the maximum length, it was truncated. Additionally, we specified to return the attention mask accordingly.

3. **Data Preparation and Training Setups**: Following the conventional workflow, we set the batch size of the dataloader to 16, and shuffle the training set to improve robustness. The initial number of epochs is set to 3, which can be adjusted as needed, we run 5 epochs. We choose AdamW as our optimizer, with the learning rate set to the commonly used 5e-5, and linearly decrease the learning rate during the training phase for fine-tuning. And Cross-entropy is used as the loss function.

4. **Feature Representation**:
Text data is converted into input IDs (`input_ids`) and attention masks (`attention_mask`), where `input_ids` represent word IDs in the vocabulary, and `attention_mask` indicates real words versus padding by a binary sequence.

5. **Classifier Architecture:**
Finally, these features are passed through a linear layer, which maps the 768-dimensional features from the BERT output to predictions for three polarity categories. The classifier also includes a mean pooling layer used to extract feature representations from the output of the model. This composite feature representation allows the model to effectively categorize sentiments in the given sentiment analysis task.

## Model Selection

- We tried out multiple models including DistilBERT, BERT, RoBERTa, ELECTRA, and DeBERTa. All selected models are from the transformer family and have been pre-trained on large text corpora.
- As for the resources,the classifier makes use of pre-trained transformer models and tokenizers provided by the Hugging Face `transformers` library.

## Reference Paper
The paper introduces an innovative approach to aspect-based sentiment analysis (ABSA), aimed at improving the ability of pre-trained language models (PLMs) like BERT to perform ABSA tasks. It proposes three aspect-oriented input transformation methods to enhance the focus of PLMs on specific aspects within sentences and also presents an adversarial benchmark (advABSA) to test the robustness of models in context modeling for aspects. 

Our model selection and architecture, including the construction of the `MeanPooling` class, are based on the methodologies from this paper to ensure more accurate capture and representation of aspect-related contextual information in sentiment analysis.
###  Results 

We conducted experiments using various models including DistilBert, ELECTRA Small Discriminator, Bert, Roberta, and Deberta.

Based on the results, although RoBERTa achieves the best performance, considering factors such as training time and accuracy, especially since the reference paper utilized BERT, we ultimately chose BERT-base-uncased as our final classifier. Below is the accuracy we obtained on the dev dataset.

| Model                           | Mean Dev Accuracy (%) | Standard Deviation (%) | Execution Time (s) | Time Per Run (s) |
|---------------------------------|-----------------------|------------------------|---------------------|------------------|
| Bert                            | 84.47                 | 0.40                   | 1929.01             | 385              |



**To run the code**

1. Open your terminal
2. Type `cd [path to src folder]`
3. Run `python tester.py`
