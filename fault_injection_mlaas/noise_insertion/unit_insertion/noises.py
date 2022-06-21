from typing import List
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
from noise_insertion.utils import return_similarity

def test_noise(noise_func, units_to_alter):
    text = "The white fox jumps over the blue wall. This is horrible."
    print("before: ", text)

    result = noise_func(text_lists=[text], aug_level=units_to_alter)

    print("after: ", result[0])
    print(return_similarity(text, result[0]))

def tokenizer(text):
    return [text]

def reverse_tokenizer(token_list):
    return ''.join(token_list).strip()

def OCR_Aug(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = nac.OcrAug(
                    aug_char_min=int(aug_level),
                    aug_char_max=int(aug_level),
                    aug_word_max=None,
                    min_char=0,
                    tokenizer=tokenizer,
                    reverse_tokenizer=reverse_tokenizer
        )

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def Keyboard_Aug(text_lists, aug_level=4) -> List[str]:
    if(int(aug_level)==0): return text_lists

    aug = nac.KeyboardAug(
                    aug_char_min=int(aug_level),
                    aug_char_max=int(aug_level),
                    aug_word_max=None,
                    min_char=0,
                    tokenizer=tokenizer,
                    reverse_tokenizer=reverse_tokenizer
        )

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def Word_swap(text_lists, aug_level=4) -> List[str]:
    if(int(aug_level)==0): return text_lists

    aug = naw.RandomWordAug(action='swap',
                            aug_min=int(aug_level), \
                            aug_max=int(aug_level))

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def Random_char_replace(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = nac.RandomCharAug(
                    action='substitute',
                    spec_char='!@#$%^&*()_+.',
                    aug_char_min=int(aug_level),
                    aug_char_max=int(aug_level),
                    aug_word_max=None,
                    min_char=0,
                    tokenizer=tokenizer,
                    reverse_tokenizer=reverse_tokenizer
        )

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def Char_swap(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = nac.RandomCharAug(action='swap',
                            swap_mode='adjacent',
                            spec_char='!@#$%^&*()_+.',
                            aug_char_min=int(aug_level),
                            aug_char_max=int(aug_level),
                            aug_word_max=None,
                            min_char=0,
                            tokenizer=tokenizer,
                            reverse_tokenizer=reverse_tokenizer
        )

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def AntonymAug(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.AntonymAug(name='Antonym_Aug', 
                         aug_min=int(aug_level), 
                         aug_max=int(aug_level),
                         lang='eng',
                         stopwords=None,
                         stopwords_regex=None,
                         verbose=1)

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

# deu erro, tenho q ver
def WordEmbsAug(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.WordEmbsAug(model_type='word2vec',
                          model_path='models/cbow_s300.txt',
                          aug_min=int(aug_level), 
                          aug_max=int(aug_level))

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def SpellingAug(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.SpellingAug(dict_path='./en.natural.txt',
                          aug_min=int(aug_level), 
                          aug_max=int(aug_level)
                          )

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def SplitAug(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.SplitAug(aug_min=int(aug_level), 
                       aug_max=int(aug_level),
                       min_char=2,
                    #    tokenizer=tokenizer,
                    #    reverse_tokenizer=reverse_tokenizer
                       )

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts


def SynonymAug(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = naw.SynonymAug(aug_min=int(aug_level), 
                         aug_max=int(aug_level))

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

def SentenceShuffle(text_lists, aug_level=4):
    if(int(aug_level)==0): return text_lists

    aug = nas.RandomSentAug(aug_min=0, # necessary to work with text with small number of sentences
                            aug_max=int(aug_level),
                            mode="random")

    augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

    return augmented_texts

# def BackTranslation(text_lists, aug_level=4):
#     aug = naw.BackTranslationAug(device='cpu',
#                                  max_length=200,
#                                  batch_size=1,
#                                  force_reload=True)

#     augmented_texts = aug.augment(text_lists, n=1, num_thread=1)

#     return augmented_texts