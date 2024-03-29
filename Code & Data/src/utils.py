""" Transform polarity into numeric target variable """
def polarity_to_numerical(polarity):
    if polarity == 'positive':
        return 0
    elif polarity == 'negative':
        return 1
    elif polarity == 'neutral':
        return 2
    else:
        return -1  # Handle unknown polarity
    
""" Map back prediction value into true polarity"""
def map_prediction_to_label(prediction):
    if prediction == 0:
        return 'positive'
    elif prediction == 1:
        return 'negative'
    elif prediction == 2:
        return 'neutral'
    else:
        return 'unknown'  # Handle any unexpected case