import nlpaug.augmenter.char as nac

def tokenizer(text):
    return [text]

def reverse_tokenizer(token_list):
    return ''.join(token_list).strip()

def OCR_Aug(text_lists, char_to_alter=4):
    aug = nac.OcrAug(
                    aug_char_min=char_to_alter,
                    aug_char_max=char_to_alter,
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

def Keyboard_Aug(text_lists, char_to_alter=4):
    aug = nac.KeyboardAug(
                    aug_char_min=char_to_alter,
                    aug_char_max=char_to_alter,
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