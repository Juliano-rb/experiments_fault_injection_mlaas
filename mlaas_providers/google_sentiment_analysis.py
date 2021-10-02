from google.cloud import language_v1
from mlaas_providers.sentiment_analysis import SentimentAnalysis
import queue
from utils.requestQueue import RequestQueue

text_type = language_v1.Document.Type.PLAIN_TEXT

def call_google_sentiment(review, client, result_queue):
    document = language_v1.Document(content=review, type_=text_type, language="EN")
    result = client.analyze_sentiment(request={'document': document})
    result_data = {'sentiment': 'positive' if result.document_sentiment.score >= 0 else 'negative'}
    result_queue.put(result_data)

class GoogleSentimentAnalysis(SentimentAnalysis):

    def __init__(self):
        self.client = language_v1.LanguageServiceClient()
        self.MAX_COMMENT_SIZE = 5000

    def classify(self, documents):
        result_queque = queue.Queue()
        request_queue = RequestQueue(function_to_call=call_google_sentiment,
                                    iterate_args=documents,
                                    fixed_args=[self.client, result_queque],
                                    call_rate=600)
        request_queue.run()

        results = []
        for i in range(len(documents)):
            results.append(result_queque.get())
        results = list(map(lambda r: r['sentiment'], results))
        
        return results
