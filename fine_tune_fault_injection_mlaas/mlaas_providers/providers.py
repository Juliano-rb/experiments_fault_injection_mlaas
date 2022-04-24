import random
import pandas as pd
from .azure_sentiment_analysis import AzureSentimentAnalysis
from .amazon_sentiment_analysis import AmazonSentimentAnalysis
from .google_sentiment_analysis import GoogleSentimentAnalysis

def azure(dataset):
    azure = AzureSentimentAnalysis()
    return azure.classify_sentiments(dataset)

def google(dataset):
    google = GoogleSentimentAnalysis()
    return google.classify(dataset)

def amazon(dataset):
    amazon = AmazonSentimentAnalysis()
    return amazon.classify(dataset)

def naive_classifier(dataset):
    possible_results =['negative', 'neutral', 'positive']
    result = []
    for i in range(len(dataset)):
        result_index = random.randint(-1, 1) 
        result.append( possible_results[result_index])

    # print(result)
    return result