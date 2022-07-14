import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.model.word_embs as nmw
from nlpaug.util.text.tokenizer import Tokenizer
import nltk


# try:
#     nltk.data.find('averaged_perceptron_tagger')
# except LookupError:
#     nltk.download('averaged_perceptron_tagger')

# try:
#     nltk.data.find('wordnet')
# except LookupError:
#     nltk.download('wordnet')

# try:
#     nltk.data.find('omw-1.4')
# except LookupError:
#     nltk.download('omw-1.4')

# uncomment on first execution
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def tokenizer(text):
    return [text]

def reverse_tokenizer(token_list):
    return ''.join(token_list).strip()

# word augmenters

def Antonym(text_lists, aug_level=0.3):
    aug = naw.AntonymAug(name='Antonym_Aug', aug_min=0, aug_max=None,
                aug_p=aug_level, lang='eng', stopwords=None,
                stopwords_regex=None, verbose=1)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def Synonym(text_lists, aug_level=0.3):
    aug = naw.SynonymAug(aug_p=aug_level,aug_min=0,aug_max=None)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def WordEmbeddings(text_lists, aug_level=0.3):
    aug = naw.WordEmbsAug(model_type='glove',
                          model_path='models/glove.twitter.27B.100d.txt',
                          aug_p=aug_level,aug_min=0,aug_max=None)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def ContextualWordEmbs(text_lists, aug_level=0.3):
    aug = naw.ContextualWordEmbsAug(aug_p=aug_level,aug_min=0,aug_max=None, verbose=True,device="cpu")

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def WordSwap(text_lists, aug_level=0.3):
    aug = naw.RandomWordAug(action='swap', aug_p=aug_level,aug_min=0,aug_max=None,verbose=True)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def Spelling(text_lists, aug_level=0.3):
    aug = naw.SpellingAug(dict_path='./models/en.natural.txt', aug_p=aug_level,aug_min=0,aug_max=None)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def WordSplit(text_lists, aug_level=0.3):
    aug = naw.SplitAug(aug_p=aug_level, aug_min=1, aug_max=1000)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

# tentar gerar o modelo usando o próprio dataset
def TfIdfWord(text_lists, aug_level=0.3):
    aug = naw.TfIdfAug(model_path='./models/tfidf', aug_p=aug_level,aug_min=0,aug_max=None)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

## apenas faz a substituição de palavras, não usarei
def ReservedAug(text_lists, aug_level=0.3):
    aug = naw.ReservedAug(reserved_tokens=[],aug_min=1, aug_max=None, aug_p=aug_level )

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def SentenceShuffle(text_lists, aug_level=0.3):
    aug = nas.RandomSentAug(aug_p=aug_level, tokenizer = None, mode="left")

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def OBackTranslation(text_lists, aug_level=0.3):
    aug = naw.BackTranslationAug(device='cpu', max_length=200, batch_size=1, force_reload=True)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts
# sentence augmenters
# gpt2? wtf
def OContextualWordEmbsForSentenceAug(text_lists, aug_level=0.3):
    aug = nas.ContextualWordEmbsForSentenceAug(min_length=0, model_type='gpt2')

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def OAbstSummAug(text_lists, aug_level=0.3):
    text_lists = [t[0:512] for t in text_lists]

    aug = nas.AbstSummAug(min_length=0)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def OLambadaAug(text_lists, aug_level=0.3):
    aug = nas.LambadaAug(model_dir='./models/lambda/out', min_length=0)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts
