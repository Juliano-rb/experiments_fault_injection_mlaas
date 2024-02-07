# Description
This repository contains the experiment performed in my master's thesis.

## Abstract
Natural Language Processing (NLP) has revolutionized industries, streamlining customer service through applications in healthcare, finance, legal, and human resources domains, and simplifying tasks like medical research, financial analysis, and sentiment analysis. To avoid the high costs of building and maintaining NLP infrastructure, companies turn to Cloud NLP services offered by major cloud providers like Amazon, Google, and Microsoft. However, there is little knowledge about how effective these services are when subjected to noise. This paper presents a study that analyzes the effectiveness of Cloud NLP services by evaluating the resistant to noise of sentiment analysis services provided by Amazon, Google, and Microsoft when subjected to 12 types of noise, including syntactic and semantic noises. The findings indicate that Google is the most effective to syntactic noises, and Microsoft is the most effective to semantic noises. These findings may help developers and companies in selecting the most suitable service provider and shed light towards improving state-of-the-art techniques for effective cloud NLP services.

# Running through Google Colab
1. Open the jupyter notebook directly in Google colab: https://colab.research.google.com/github/evaluating-effectiveness-cloud-nlp/replication_package/blob/master/notebook_rq1.ipynb.
2. Follow notebook instructions.

# Running locally (not recomended)
## Prerequisites
- `python 3`.
- `pipenv` installed [``how to install``](https://pipenv.pypa.io/en/latest/#install-pipenv-today).
- to run with real MlaaS providers, you should provide valid credentials to each used provider:
    - create a `credentials.py` file following the structure of `credentials.example.py`.
    - fill the credentials to Amazon, Microsoft and Google providers.
## Running experiments
1. activate the environmentment: ``pipenv shell``.
2. install dependencies: ``pipenv sync``. The process may take several minutes.
2. with the environmentment activated, run the notebook: 
    1. jupyter-lab with command ``jupyter-lab``.
    2. open the file `notebook_experiment.ipynb`.
    3. follow notebook's instructions.

## Hints
- you don't need to download the `glove.twitter` word embedding model if you remove the noise algorithm `WordEmbeddings` from `noise_list` variable in both experiments.
- by default, the MLaaS providers are mocked with random predictions, so you can easily try the algorithm, but in a real use you should register an account in each one and fill the `creditials.py` file.
- in both experiments, in case of error (eg. network error) is possible to continue from a previously running experiment by filling the `Continue from` text area in the notebook with a `/outputs` folder.
- is possible to manipulate the `progress.json` file in order to re-run some task of the experiment.
