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

In questo script sono presenti due layer:
* il primo layer riceve un batch (gruppo) in input.
* il secondo layer ottiene l'output del primo layer e calcola a sua volta l'output

Lo script e' praticamente identico allo script "04.1_batch_input.py",
sono state duplicate e modificate le variabili dei pesi e dei bias.

"""

__author__ = "Zenaro Stefano"
__version__ = "01_01 2020-05-03"

import utils  # per funzioni di test

# possiamo richiamare oggetti (funzioni,...) numpy facendo np.x(),...
import numpy as np

boold = False  # per debug

inputs = [
    [1, 2, 3, 2.5],         # set 1 di input
    [2.0, 5.0, -1.0, 2.0],  # set 2 di input
    [-1.5, 2.7, 2.2, -0.8]  # set 3 di input
]

# pesi del layer 1
weights = [
    # pesi collegamenti verso il neurone 1
    [
        0.2,   # peso collegamento input 1 - neurone 1 (layer 1)
        0.8,   # peso collegamento input 2 - neurone 1 (layer 1)
        -0.5,  # peso collegamento input 3 - neurone 1 (layer 1)
        1.0    # peso collegamento input 4 - neurone 1 (layer 1)
    ],

    # pesi collegamenti verso il neurone 2
    [
        0.5,    # peso collegamento input 1 - neurone 2 (layer 1)
        -0.91,  # peso collegamento input 2 - neurone 2 (layer 1)
        0.26,   # peso collegamento input 3 - neurone 2 (layer 1)
        -0.5    # peso collegamento input 4 - neurone 2 (layer 1)
    ],

    # pesi collegamenti verso il neurone 3
    [
        -0.26,  # peso collegamento input 1 - neurone 3 (layer 1)
        -0.27,  # peso collegamento input 2 - neurone 3 (layer 1)
        0.17,   # peso collegamento input 3 - neurone 3 (layer 1)
        0.87    # peso collegamento input 4 - neurone 3 (layer 1)
    ]
]

# errori statistici layer 1
biases = [2,   # errore statistico neurone 1 (layer 1)
          3,   # errore statistico neurone 2 (layer 1)
          0.5  # errore statistico neurone 3 (layer 1)
          ]

# pesi del layer 2
weights2 = [
    [
        0.1,    # peso collegamento neurone 1 (layer 1) - neurone 1 (layer 2)
        -0.14,  # peso collegamento neurone 2 (layer 1) - neurone 1 (layer 2)
        0.5,    # peso collegamento neurone 3 (layer 1) - neurone 1 (layer 2)
    ],

    # pesi collegamenti verso il neurone 2
    [
        -0.5,   # peso collegamento neurone 1 (layer 1) - neurone 2 (layer 2)
        0.12,   # peso collegamento neurone 2 (layer 1) - neurone 2 (layer 2)
        -0.33,  # peso collegamento neurone 3 (layer 1) - neurone 2 (layer 2)
    ],

    # pesi collegamenti verso il neurone 3
    [
        -0.44,  # peso collegamento neurone 1 (layer 1) - neurone 3 (layer 2)
        0.73,   # peso collegamento neurone 2 (layer 1) - neurone 3 (layer 2)
        -0.13,  # peso collegamento neurone 3 (layer 1) - neurone 3 (layer 2)
    ]
]

# errori statistici layer 2
biases2 = [
    -1,   # errore statistico neurone 1 (layer 2)
    2,    # errore statistico neurone 2 (layer 2)
    -0.5  # errore statistico neurone 3 (layer 2)
]


if __name__ == "__main__":
    # prodotto interno tra input e pesi dei neuroni del livello 1 + errori statistici livello 1:
    # output dei neuroni del layer 1
    layer1_output = np.dot(inputs, np.array(weights).T) + biases

    # prodotto interno tra output del layer1 e pesi dei neuroni del livello 2 + errori statistici livello 2:
    # output dei neuroni del layer 2
    layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

    if boold:
        print("Output livello 1:")
        print(layer1_output)
        print("Output livello 2:")
    
    print(layer2_output)

    # -- SEZIONE DI TEST

    # CALCOLA OUTPUT DEL LAYER 1
    # output dei neuroni del livello, per ogni sample/set
    desired_outputs_layer1 = utils.calc_batches_layer_by_props(inputs, weights, biases)
    
    # errore se il risultato e' diverso da quello desiderato
    assert np.allclose(layer1_output, desired_outputs_layer1)

    # CALCOLA OUTPUT DEL LAYER 2
    # output dei neuroni del livello, per ogni sample/set
    desired_outputs_layer2 = utils.calc_batches_layer_by_props(desired_outputs_layer1, weights2, biases2)

    # errore se il risultato e' diverso da quello desiderato
    assert np.allclose(layer2_output, desired_outputs_layer2)
