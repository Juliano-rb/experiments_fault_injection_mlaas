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

def sentiment_analysis_with_opinion_mining(client, documents):
    result = client.analyze_sentiment(documents, show_opinion_mining=True)
    # print(result)
    doc_result = [doc for doc in result if not doc.is_error]

    sentiments = ['positive' if d.confidence_scores.positive > d.confidence_scores.negative else 'negative'  for d in doc_result] 
    return sentiments
          
# chama o servico de classificação azure em grupos de 10 em 10
def classify_sentiments(documents):
    client = authenticate_client()

    chunks = []
    results = []
    for i in range(0, len(documents), 10):
        chunk = documents[i:i+10]
        chunks.append(chunk)

    count = 0
    for c in chunks:
        result = sentiment_analysis_with_opinion_mining(client, c)
        # print(count,'/', len(documents),'...\r', sep=None)
        results.extend(result)
        count += len(c)
        # sleep(5)

    print("\n")
    return results