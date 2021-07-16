import azure_sentiment_analysis
import random

# microsoft
def azure(dataset):
    return azure_sentiment_analysis.classify_sentiments(dataset)


def amazon(dataset):
    return azure_sentiment_analysis.classify_sentiments(dataset)

def naive_classifier(dataset):
    result = []
    for i in range(len(dataset)):
        result.append(random.randint(0, 1))

    # print(result)
    return result