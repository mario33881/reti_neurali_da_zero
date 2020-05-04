"""
# Video 4: Video 4: Batch, livelli e programmazione ad oggetti

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

Questo script ottiene lo stesso risultato
dello script "04.2_due_layer.py" attraverso
la programmazione ad oggetti.

Inoltre i valori di inizializzazione
degli errori statistici e dei pesi
sono realistici e non piu' valori inventati.
"""

__author__ = "Zenaro Stefano"
__version__ = "01_01 2020-05-03"

import utils  # per funzioni di test

# possiamo richiamare oggetti (funzioni,...) numpy facendo np.x(),...
import numpy as np

boold = False  # per debug

# input del primo layer
X = [
    [1, 2, 3, 2.5],         # set 1 di input
    [2.0, 5.0, -1.0, 2.0],  # set 2 di input
    [-1.5, 2.7, 2.2, -0.8]  # set 3 di input
]


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        # chiediamo al programmatore il numero di input in un set del batch e di neuroni
        # per creare una matrice di pesi di <n_inputs> righe e <n_neurons> colonne
        # Moltiplichiamo i valori per 0.10 per ridurre il valore dei pesi
        # > randn(): distribuzione gaussiana attorno a 0
        # > e i suoi parametri producono la shape
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        # creiamo una "matrice" di 0 con una riga e tante colonne/zeri quanti sono i neuroni
        # > occorre passare la shape come array: (x, y)
        self.biases = np.zeros((1, n_neurons))

        # definisco output nel metodo __init__ per PEP8
        self.output = None

    def forward(self, inputs):
        # <inputs> e' l'output del layer precedente
        self.output = np.dot(inputs, self.weights) + self.biases


if __name__ == "__main__":

    # imposta seed per generare gli stessi valori con il random
    np.random.seed(0)

    # creo il layer 1
    # 4 = numero di input, 5 = numero di neuroni e quindi di output
    layer1 = LayerDense(4, 5)

    # creo il layer 2
    # 5 = input dal layer precedente (5 neuroni), 2 = numero di neuroni e quindi di output
    layer2 = LayerDense(5, 2)

    # passo gli input al layer1
    layer1.forward(X)
    layer1_output = layer1.output  # output del layer 1

    # pass l'output del layer 1 al layer 2
    layer2.forward(layer1_output)
    layer2_output = layer2.output  # output del layer 2

    if boold:
        print("Output:")
    print(layer2_output)

    # -- SEZIONE DI TEST
    np.random.seed(0)

    # CALCOLA OUTPUT DEL LAYER 1
    # output dei neuroni del livello, per ogni sample/set
    desired_outputs_layer1 = utils.calc_batches_layer_by_props(X, np.array(layer1.weights).T, layer1.biases[0])

    # errore se il risultato e' diverso da quello desiderato
    assert np.allclose(layer1_output, desired_outputs_layer1)

    # CALCOLA OUTPUT DEL LAYER 2
    # output dei neuroni del livello, per ogni sample/set
    desired_outputs_layer2 = utils.calc_batches_layer_by_props(desired_outputs_layer1,
                                                               np.array(layer2.weights).T,
                                                               layer2.biases[0])

    if boold:
        print("Desired output:")
        print(desired_outputs_layer2)

    # errore se il risultato e' diverso da quello desiderato
    assert np.allclose(layer2_output, desired_outputs_layer2)
