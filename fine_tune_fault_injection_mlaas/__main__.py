from data_sampling import DataSampling
from noise import OCR_Aug
from mlaas_providers import providers as ml_providers

if __name__ == '__main__':
    dataSampling = DataSampling('Tweets_dataset.csv')

    data = dataSampling.get_by_width(100, 15, 20)

    noised = OCR_Aug(data["text"], char_to_alter=10)

    ml_providers.amazon

    print("Original: ", data["text"])
    print("Noised: ",noised)
     