"""
# Video 5: Activation function

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

Questo script utilizza l'oggetto LayerDense()
per creare un layer e calcolare il risultato.

Il risultato del layer viene passato attraverso
l'oggetto ActivationReLU() per calcolare il risultato 
della funzione matematica "rectified linear function" (rettificatore)

Ora viene usata la libreria nnfs per garantire
che i risultati di numpy siano sempre uguali, per motivi didattici,
e per permettere di generare dei dataset.
"""

__author__ = "Zenaro Stefano"
__version__ = "01_01 2020-05-14"

import utils  # per funzioni di test

# possiamo richiamare oggetti (funzioni,...) numpy facendo np.x(),...
import numpy as np

# importa nnfs per usare nnfs.init() e garantire cosi' i risultati
import nnfs
# importa la funzione che permette di generare i dataset di tipo spirale
from nnfs.datasets import spiral_data

boold = False  # per debug


class ActivationReLU:

    def __init__(self):
        # definisco output nel metodo __init__ per PEP8
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


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

    # inizializza nnfs per garantire gli stessi risultati di numpy
    # > usato per motivi didattici
    nnfs.init()

    # genero il dataset: sono 100 sample, 3 classi, ogni sample ha 2 feature (x e y di un punto)
    # X e' sono i feature set; y sono i label, target o classifications (classi)
    X, y = spiral_data(100, 3)

    # creo il layer 1
    # 2 = numero di input (i punti hanno coordinate x e y), 5 = numero di neuroni e quindi di output
    layer1 = LayerDense(2, 5)

    # passo gli input al layer1
    layer1.forward(X)
    layer1_output = layer1.output  # output del layer 1

    # creo oggetto della activation function
    activation1 = ActivationReLU()

    # passo l'output del layer all'activation function:
    # il suo output (activation1.output) contiene solo valori 0 o maggiori di 0
    activation1.forward(layer1_output)
    activation1_output = activation1.output  # output activation function

    if boold:
        print("output:")
    print(activation1_output)

    # -- SEZIONE DI TEST

    # CALCOLA OUTPUT DEL LAYER 1
    # output dei neuroni del livello, per ogni sample/set
    desired_outputs_layer1 = utils.calc_batches_layer_by_props(X, np.array(layer1.weights).T, layer1.biases[0])

    # errore se il risultato e' diverso da quello desiderato
    assert np.allclose(layer1_output, desired_outputs_layer1)

    # CALCOLA OUTPUT DEL LAYER DOPO ESSERE PASSATO PER LA FUNZIONE RETTIFICATORE
    desired_relu_outputs_layer1 = utils.batch_layer_relu_function(desired_outputs_layer1)

    if boold:
        print("desired output:")
        print(desired_relu_outputs_layer1)

    # errore se il risultato e' diverso da quello desiderato
    assert np.allclose(activation1_output, desired_relu_outputs_layer1)
