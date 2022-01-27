from os import sep
from noise_insertion import noises

text = "The few scenes that actually attempt a depiction of revolutionary struggle resemble a hirsute Boy Scout troop meandering tentatively between swimming holes."
noise_level = 0.3

noise_algorithms=[
    noises.keyboard_aug,
    noises.ocr_aug,
    noises.random_noise,
    noises.char_swap_noise,
    noises.aug.AntonymAug,
    noises.aug.RandomWordAug,
    noises.aug.SpellingAug,
    noises.aug.SplitAug,
    noises.aug.SynonymAug,
    noises.aug.TfldfAug,
    noises.aug.ReservedAug,
    noises.aug.AbstSummAug,
    noises.aug.RandomSentAug,
    noises.aug.WordEmbsAug,
    noises.aug.ContextualWordEmbsAug,
    noises.aug.ContextualWordEmbsForSentenceAug,
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