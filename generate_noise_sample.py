from os import sep
from noise_insertion import noise_insertion

text = "The few scenes that actually attempt a depiction of revolutionary struggle resemble a hirsute Boy Scout troop meandering tentatively between swimming holes."
noise_level = 0.3

noise_algorithms=[
    noise_insertion.keyboard_aug,
    noise_insertion.ocr_aug,
    noise_insertion.random_noise,
    noise_insertion.char_swap_noise,
    noise_insertion.aug.AntonymAug,
    noise_insertion.aug.RandomWordAug,
    noise_insertion.aug.SpellingAug,
    noise_insertion.aug.SplitAug,
    noise_insertion.aug.SynonymAug,
    noise_insertion.aug.TfldfAug,
    noise_insertion.aug.ReservedAug,
    noise_insertion.aug.AbstSummAug,
    noise_insertion.aug.RandomSentAug,
    noise_insertion.aug.WordEmbsAug,
    noise_insertion.aug.ContextualWordEmbsAug,
    noise_insertion.aug.ContextualWordEmbsForSentenceAug,
    # noise_insertion.aug.BackTranslation, # error
    # noise_insertion.aug.LambadaAug # error
]

noise_results = []

print(text)
for noise in noise_algorithms:
    print(noise.__name__,'...',end='')
    noised_text = noise([text], noise_level)
    print("\r-",noise.__name__, " : ", noised_text)
    noise_results.append({'noise': noise.__name__, 'result': noised_text})

print(text)
print(noise_results)