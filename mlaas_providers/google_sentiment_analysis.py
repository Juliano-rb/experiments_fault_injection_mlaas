from google.cloud import language_v1

from time import sleep

from mlaas_providers.sentiment_analysis import SentimentAnalysis

import queue
import threading, time, random

text_type = language_v1.Document.Type.PLAIN_TEXT

def call_google_sentiment(client, review, index, result_queue):
    document = language_v1.Document(content=review, type_=text_type, language="EN")
    result = client.analyze_sentiment(request={'document': document})
    result_queue.put({'sentiment': 'positive' if result.document_sentiment.score >= 0 else 'negative', 'index': index})

class GoogleSentimentAnalysis(SentimentAnalysis):

    def __init__(self):
        self.client = language_v1.LanguageServiceClient()
        self.MAX_COMMENT_SIZE = 5000

    def classify(self, documents):
        batches = self.chunks(documents, 600)
        results = []
        first = True
        for b in batches:
            if not first:
                print('Waiting for 70 seconds due to 600/min maximum requests')
                sleep(70)
            q = queue.Queue()
            threads = [threading.Thread(target=call_google_sentiment, args=(self.client, r, i, q)) for i, r in enumerate(b)]
            for t in threads:
                t.daemon = True
                t.start()
            ans = []
            for i in range(len(b)):
                ans.append(q.get())
            ans.sort(key=lambda r: r['index'])
            ans = list(map(lambda r: r['sentiment'], ans))
            results.extend(ans)
            first = False
        return results
