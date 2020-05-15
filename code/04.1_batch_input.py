"""
# Video 4: Batch, livelli e programmazione ad oggetti

Appunti e traduzione della serie "Neural Networks from Scratch"
su Youtube di sentdex:
[clicca qui per accedere al suo canale Youtube](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)

Per supportare l'autore considerare l'acquisto del
libro dal sito: https://nnfs.io/

> Non sono associato a Sentdex, il suo canale, il suo sito e il suo libro

Notes and translation of the "Neural Networks from Scratch" series.
It's on Youtube and the author is sentdex:
[click here to go to his Youtube channel](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)

To support the author consider buying the book
at this website: https://nnfs.io/

> I'm not associated to Sentdex, his channel, his website, his book in any way, shape or form

---

In questo script si manda in input al layer
un batch (gruppo) di input.

Questo permette di parallelizzare i calcoli
e di far imparare correttamente come calcolare il risultato alla rete neurale:
pochi set alla volta rendono la rete imprecisa (si adatta solo ad un set),
troppi set alla volta rendono la rete "troppo precisa" (la rete "sa a memoria" il risultato).

Nel momento in cui ho una matrice di input devo
assicurarmi che la shape sia adatta a poter
fare il prodotto interno con la matrice dei pesi.

Se ho 3 set di 4 dati in input, shape (3,4),
devo poter moltiplicare i 4 dati per i 4 pesi, i quali hanno shape (3,4):
devo invertire le righe e le colonne dei pesi per omogeneizzare
i pesi con i dati:
* dati con shape (3,4)
* pesi con shape (4,3)

In questo modo posso fare il prodotto interno tra i 4 dati
e i 4 pesi.

Per fare lo scambio righe/colonne dei pesi occorre
convertire la lista dei pesi in un array numpy
e usare il metodo T (transpose).
> Documentazione di transpose:
> https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html?highlight=t#numpy.ndarray.T
"""

__author__ = "Zenaro Stefano"
__version__ = "01_01 2020-05-03"

import utils  # per funzioni di test

# possiamo richiamare oggetti (funzioni,...) numpy facendo np.x(),...
import numpy as np

boold = False  # variabile debug

inputs = [
    [1, 2, 3, 2.5],         # set/sample 1 di input
    [2.0, 5.0, -1.0, 2.0],  # set/sample 2 di input
    [-1.5, 2.7, 2.2, -0.8]  # set/sample 3 di input
]

weights = [
    # pesi collegamento verso il neurone 1
    [
        0.2,   # peso collegamento input 1 - neurone 1
        0.8,   # peso collegamento input 2 - neurone 1
        -0.5,  # peso collegamento input 3 - neurone 1
        1.0    # peso collegamento input 4 - neurone 1
    ],

    # pesi collegamenti verso il neurone 2
    [
        0.5,    # peso collegamento input 1 - neurone 2
        -0.91,  # peso collegamento input 2 - neurone 2
        0.26,   # peso collegamento input 3 - neurone 2
        -0.5    # peso collegamento input 4 - neurone 2
    ],

    # pesi collegamenti verso il neurone 3
    [
        -0.26,  # peso collegamento input 1 - neurone 3
        -0.27,  # peso collegamento input 2 - neurone 3
        0.17,   # peso collegamento input 3 - neurone 3
        0.87    # peso collegamento input 4 - neurone 3
    ]
]

# errori statistici
biases = [2,   # errore statistico neurone 1
          3,   # errore statistico neurone 2
          0.5  # errore statistico neurone 3
          ]

if __name__ == "__main__":
    
    # converto la lista di pesi in un array numpy
    numpy_weights = np.array(weights)

    # scambio righe e colonne (eseguo "transpose")
    # per evitare l'errore di shape
    transposed_weights = numpy_weights.T
    
    # prodotto interno tra matrici input e pesi (righe e colonne scambiate)
    output = np.dot(inputs, transposed_weights) + biases

    if boold:
        print("Output:")
    print(output)

    # -- SEZIONE DI TEST

    # calcola gli output desiderati dal layer
    desired_outputs = utils.calc_batches_layer_by_props(inputs, weights, biases)
    
    # errore se il risultato e' diverso da quello desiderato
    assert np.allclose(output, desired_outputs)
