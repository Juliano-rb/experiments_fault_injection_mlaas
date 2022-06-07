from typing import List
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from metrics.metrics import return_similarity

def test_noise(noise_func, text, units_to_alter):
    text = "the white fox jumps over the blue wall"

    result = noise_func(text_lists=[text], unit_to_alter=units_to_alter)

    print(result[0])
    print(return_similarity(text, result[0]))

def tokenizer(text):
    return [text]

def reverse_tokenizer(token_list):
    return ''.join(token_list).strip()

def OCR_Aug(text_lists, unit_to_alter=4):
    aug = nac.OcrAug(
                    aug_char_min=unit_to_alter,
                    aug_char_max=unit_to_alter,
                    # aug_char_p=None,
                    # aug_word_p=None,
                    # aug_word_min=0,
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

def Keyboard_Aug(text_lists, unit_to_alter=4) -> List[str]:
    aug = nac.KeyboardAug(
                    aug_char_min=unit_to_alter,
                    aug_char_max=unit_to_alter,
                    # aug_char_p=None,
                    # aug_word_p=None,
                    # aug_word_min=0,
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

def Word_swap(text_lists, unit_to_alter=4) -> List[str]:
    aug = naw.RandomWordAug(action='swap', aug_min=unit_to_alter, \
                            aug_max=unit_to_alter)

    augmented_texts = []

    for text in text_lists:
        augmented_text = aug.augment(text, n=1)
        augmented_texts.append(augmented_text)

    return augmented_texts

def Random_char_replace(text_lists, unit_to_alter=4):
    aug = nac.RandomCharAug(
                    action='substitute',
                    spec_char='!@#$%^&*()_+.', #checar se inclui o ponto
                    aug_char_min=unit_to_alter,
                    aug_char_max=unit_to_alter,
                    # aug_char_p=None,
                    # aug_word_p=None,
                    # aug_word_min=0,
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