## Prerequisites
- `python 3`.
- `pipenv` installed [``how to install``](https://pipenv.pypa.io/en/latest/#install-pipenv-today).

## Running experiments
1. activate the envrioment: ``pipenv shell``.
2. install dependencies: ``pipenv install``. The process may take several minutes.
2. with the envrioment activated, run the notebook: 
    1. jupyter-lab with command ``jupyter-lab``
    2. follow notebook's instructions.

## Hints
- you don't need to download the `glove.twitter` word embedding model if you remove the noise algorithm `WordEmbeddings` from `noise_list` variable in both experiments.
- by default, the MLaaS providers are mocked with random predictions, so you can easily try the algorithm, but in a real use you should register an account in each one and fill the `creditials.py` file.
- in both experiments, in case of error (eg. network error) is possible to continue from a previously running experiment by filling the `Continue from` text area in the notebook with a `/outputs` folder.
- is possible to manipulate the `progress.json` file in order to re-run some task of the experiment.