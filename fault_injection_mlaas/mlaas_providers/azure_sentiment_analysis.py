from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import sys
from time import sleep
import credentials
import queue
from utils.requestQueue import RequestQueue

def map_sentiment(result):
    if(result.sentiment=='negative'):
        return 'negative'
    elif result.sentiment=='positive':
        return 'positive'
    else:
        return 'neutral'

class AzureSentimentAnalysis:
    def __init__(self):
        self.azure_text_analytics = self.authenticate_client()

        self.MAX_COMMENT_SIZE = 5000

    def authenticate_client(self):
        endpoint = credentials.azure_endpoint
        key = credentials.azure_key_1

        ta_credential = AzureKeyCredential(key)
        text_analytics_client = TextAnalyticsClient(
                endpoint=endpoint, 
                credential=ta_credential)
        return text_analytics_client

    def sentiment_analysis_with_opinion_mining(self, data, client, result_queue):
        try:
            print('\r'+str(len(result_queue.queue)), end='')
            result = client.analyze_sentiment(data["documents"], show_opinion_mining=True)
            # print(result)
            doc_result = [doc for doc in result if not doc.is_error]

            sentiments = list(map(map_sentiment, doc_result))

            result_queue.put({"predictions":sentiments, "index":data["index"]})
        except Exception as e:
            print("Error:", e)
            print(data["documents"])
            raise e

    def arrange_results(self, result_list):
        result_list.sort(key=lambda x: x["index"], reverse=False)

        data = list(map(lambda x: x["predictions"], result_list))
        data = [item for sublist in data for item in sublist]

        return data

    # chama o servico de classificação azure em grupos de 10 em 10
    def classify_sentiments(self, documents, call_rate_param=30):
        chunks = []
        results = []
        for i in range(0, len(documents), 10):
            chunk = documents[i:i+10]
            chunks.append({"documents":chunk, "index": i})

        result_queque = queue.Queue()
        request_queue = RequestQueue(function_to_call=self.sentiment_analysis_with_opinion_mining,
                                    iterate_args=chunks,
                                    fixed_args=[self.azure_text_analytics, result_queque],
                                    call_rate=call_rate_param)
        request_queue.run()
        if len(result_queque.queue) != len(chunks):
            raise "error in one of the threads"

        print('\r', end='')

        results = []
        while len(result_queque.queue)>0:
            # for i in range(len(documents)):
            data = result_queque.get()
            results.append(data)

        arranged = self.arrange_results(results)

        return arranged