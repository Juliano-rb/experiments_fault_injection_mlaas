from time import sleep

import boto3
import credentials
from mlaas_providers.sentiment_analysis import SentimentAnalysis


class AmazonSentimentAnalysis(SentimentAnalysis):

    def __init__(self):
        self.comprehend = boto3.client(service_name='comprehend',
                                       region_name='us-east-1',
                                       aws_access_key_id=credentials.aws_access_key_id,
                                       aws_secret_access_key=credentials.aws_secret_access_key)

        self.MAX_COMMENT_SIZE = 5000

    def call_service(self, batch):
        batch = self.ensure_limits(batch)
        result = self.comprehend.batch_detect_sentiment(TextList=batch, LanguageCode='en')
        result = list(map(lambda r: 'positive' if r['SentimentScore']['Positive'] > r['SentimentScore']['Negative'] else 'negative', result['ResultList']))
        return result

    def classify(self, documents):
        batches = self.chunks(documents, 10)
        results = []
        for b in batches:
            results.extend(self.call_service(b))
            sleep(1)
        return results
