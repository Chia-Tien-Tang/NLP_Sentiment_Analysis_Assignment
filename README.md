# NLP_Sentiment_Analysis_Assignment

**To run the code**
1. Open your terminal
2. type cd [path to src folder]
3. python tester.py


## Contributors

The following students contributed to this deliverable:

- Chia Tien Tang
- Zheng Wan
- Weijing Zeng
- Killian Le Metayer

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
- Each sentence in the dataset may correspond to multiple aspect categories, each with potentially different polarity evaluations.
- The same sentence may appear multiple times in the dataset due to different aspects or target terms, each time potentially having a different polarity evaluation.
- If a review contains a negative sentiment for a given aspect, other aspects will most likely be rated negative as well.