## Um novo experimento (disciplina)

1. Calcular o tamanho da sentença.
2. Aplicar com 1 noise, 2 noise, ..., 10 noise

Começar com um unico ruído.
Ex.: Testar OCR com 1, 2, ..., 10 noise.
Selecionar tamanhos especificos de sentenças e verificar o resultado

O objetivo é deixar mais realistico, no mundo real existem menos erros.


## Passo a passo:
- [x] Criar função que gera novos datasets com tamanhos especificos de sentenças
    - Intervalos: 5 à 10, 10 à 15, 15 à 20
    - Fazer um sampling balanceado utilizando as classes existentes

- [x] Criar funções que aplicam ruído a partir de uma quantidade, não uma porcentagem.

- [ ] Fazer a requisição para os serviços verificando o desempenho com e sem ruido
    - Reaproveitar as classes que executam requisições aos provedores.
    - Tentar reaproveitar algo do pipeline para gerar o experimento e gerar resultados.

## Usefull commands:
1. install a new dependency: pipenv install openpyxl
2. run project: ``pipenv run pip .`` or ``pipenv shell`` to activate envrioment and ``py .``


Mais um ruído
Pelo menos mais um provedor
Plotar os graficos de forma unida
Desacoplar um pouco

Resolver o problema com outros provedores e aumentar um pouco o cenário.