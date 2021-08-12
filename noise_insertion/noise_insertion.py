
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import string
import random
from random import randrange
from nlpaug.util import Action
from .custom_tokenizer import Tokenizer as custom_tokenizer

import os

def keyboard_aug(text_lists,aug_level=0.3):
    augmented_texts = []

    # aug = nac.KeyboardAug(aug_char_p=aug_level, aug_word_max=None, aug_word_p=1)
    aug = nac.KeyboardAug(aug_char_p=aug_level,
                      aug_char_max=None,
                      tokenizer = custom_tokenizer.tokenizer,
                      reverse_tokenizer=custom_tokenizer.reverse_tokenizer)

    for text in text_lists:
        size = len(text)

        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)
    
    # print(augmented_texts)
    return augmented_texts


def ocr_aug(text_lists, aug_level=0.3):
    aug = nac.OcrAug(aug_char_p=aug_level,
                    aug_char_max=None,
                    tokenizer = custom_tokenizer.tokenizer,
                    reverse_tokenizer=custom_tokenizer.reverse_tokenizer)

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

        # return_similarity(text, augmented_text)
    
    # print(augmented_texts)
    return augmented_texts

def random_noise(text_lists,aug_level=0.3):
    aug = nac.RandomCharAug(aug_char_p=aug_level,
                    aug_char_max=None,
                    tokenizer = custom_tokenizer.tokenizer,
                    reverse_tokenizer=custom_tokenizer.reverse_tokenizer)

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def no_noise(text_lists, aug_level=0):
    return text_lists

def return_similarity(a,b):
    size = len(a) if len(a) > len(b) else len(b)
    a = a.ljust(size)
    b = b.ljust(size)
    # print(len(a), '-', len(b), '=', size)
    equals = 0
    for i in range(size):
        if(a[i]==b[i]):
            equals+=1
    
    print("equals=",str(equals))
    print("diff=", size-equals)
    print("percent=",str(equals/size))
    return equals/size
# texts =["holy fuck, this is incredible.", "thanks mom, i like to drink this"]

# ocr_aug(texts)




# a = 'Hello world my name is berry allen'
# a_joined = a.split(' ')
# a_joined = '_'.join(a_joined)
# print(a_joined)

# percent = 0.20
# chars = round(percent*len(a_joined))
# aug = nac.KeyboardAug(aug_char_p=0.1, aug_char_min=1, aug_char_max=None, aug_word_min=1, aug_word_max=1)

# aug = nac.RandomCharAug(aug_char_min=5, aug_char_max=5)

# b = aug.augment(a_joined, n=1)
# print(b)

# b.split('-').join(' ')

# b = random_noise()
# print(a)
# print(b)
# return_similarity(a,b)