from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import sys
from time import sleep
import credentials


def authenticate_client():
    endpoint = credentials.azure_endpoint
    key = credentials.azure_key_1

    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=ta_credential)
    return text_analytics_client

# def sentiment_analysis_example(client):

#     documents = ["I had the best day of my life. I wish you were there with me."]
#     response = client.analyze_sentiment(documents=documents)[0]
#     print("Document Sentiment: {}".format(response.sentiment))
#     print("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
#         response.confidence_scores.positive,
#         response.confidence_scores.neutral,
#         response.confidence_scores.negative,
#     ))
#     for idx, sentence in enumerate(response.sentences):
#         print("Sentence: {}".format(sentence.text))
#         print("Sentence {} sentiment: {}".format(idx+1, sentence.sentiment))
#         print("Sentence score:\nPositive={0:.2f}\nNeutral={1:.2f}\nNegative={2:.2f}\n".format(
#             sentence.confidence_scores.positive,
#             sentence.confidence_scores.neutral,
#             sentence.confidence_scores.negative,
#         ))

def sentiment_analysis_with_opinion_mining_example(client, documents):
    result = client.analyze_sentiment(documents, show_opinion_mining=True)
    # print(result)
    doc_result = [doc for doc in result if not doc.is_error]

    sentiments = ['positive' if d.confidence_scores.positive > d.confidence_scores.negative else 'negative'  for d in doc_result] 
    return sentiments

    # for document in doc_result:
    #     print("Document Sentiment: {}".format(document.sentiment))
    #     print("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
    #         document.confidence_scores.positive,
    #         document.confidence_scores.neutral,
    #         document.confidence_scores.negative,
    #     ))
        # for sentence in document.sentences:
        #     print("Sentence: {}".format(sentence.text))
            # print("Sentence sentiment: {}".format(sentence.sentiment))
            # print("Sentence score:\nPositive={0:.2f}\nNeutral={1:.2f}\nNegative={2:.2f}\n".format(
            #     sentence.confidence_scores.positive,
            #     sentence.confidence_scores.neutral,
            #     sentence.confidence_scores.negative,
            # ))
            # for mined_opinion in sentence.mined_opinions:
            #     target = mined_opinion.target
            #     print("......'{}' target '{}'".format(target.sentiment, target.text))
            #     print("......Target score:\n......Positive={0:.2f}\n......Negative={1:.2f}\n".format(
            #         target.confidence_scores.positive,
            #         target.confidence_scores.negative,
            #     ))
            #     for assessment in mined_opinion.assessments:
            #         print("......'{}' assessment '{}'".format(assessment.sentiment, assessment.text))
            #         print("......Assessment score:\n......Positive={0:.2f}\n......Negative={1:.2f}\n".format(
            #             assessment.confidence_scores.positive,
            #             assessment.confidence_scores.negative,
            #         ))
            # print("\n")
        # print("\n")
          
def classify_sentiments(documents):
    client = authenticate_client()

    chunks = []
    results = []
    for i in range(0, len(documents), 10):
        chunk = documents[i:i+10]
        chunks.append(chunk)

    count = 0
    for c in chunks:
        result = sentiment_analysis_with_opinion_mining_example(client, c)
        # print(count,'/', len(documents),'...\r', sep=None)
        results.extend(result)
        count += len(c)
        # sleep(5)

    print("\n")
    return results