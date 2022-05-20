import random
from typing import List
import pandas as pd
from .azure_sentiment_analysis import AzureSentimentAnalysis
from .amazon_sentiment_analysis import AmazonSentimentAnalysis
from .google_sentiment_analysis import GoogleSentimentAnalysis
import types
import functools

def microsoft(dataset):
    azure = AzureSentimentAnalysis()
    return azure.classify_sentiments(dataset)

def google(dataset):
    google = GoogleSentimentAnalysis()
    return google.classify(dataset)

def amazon(dataset):
    amazon = AmazonSentimentAnalysis()
    return amazon.classify(dataset)


def run_naive_classifier(dataset, classes=['negative', 'neutral', 'positive']):
    result = []
    for i in range(len(dataset)):
        result_index = random.randint(-1, 1) 
        result.append( classes[result_index])

    # print(result)
    return result
def return_mock_of(provider):
    copy_func = types.FunctionType(run_naive_classifier.__code__,
                            run_naive_classifier.__globals__,
                            name=provider.__name__+'_mock',
                            argdefs=run_naive_classifier.__defaults__,
                            closure=run_naive_classifier.__closure__)
    copy_func = functools.update_wrapper(copy_func, run_naive_classifier)
    copy_func.__name__ = provider.__name__+'_mock'
    copy_func.__kwdefaults__ = run_naive_classifier.__kwdefaults__
    return copy_func