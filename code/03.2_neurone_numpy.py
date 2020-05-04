"""
# Video 3: prodotto interno (dot product)

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

Il codice si occupa di calcolare
l'output di un neurone (come nel video 1)
utilizzando la libreria numpy per calcolare
il prodotto interno (dot product).

Il prodotto interno calcola il prodotto
tra i singoli elementi di due liste dello stesso indice
> elemento 0 * elemento 0, elemento 1 * elemento 1,...
e somma i prodotti insieme.

Viene usato il prodotto interno per calcolare
la moltiplicazione tra input e pesi e sommare i risultati.

Al prodotto interno occorre sommare l'errore statistico (bias)

Documentazione di numpy.dot():
https://numpy.org/doc/stable/reference/generated/numpy.dot.html?highlight=dot#numpy.dot
"""

__author__ = "Zenaro Stefano"
__version__ = "01_01 2020-04-28"

import utils  # per funzioni di test

# possiamo richiamare oggetti (funzioni,...) numpy facendo np.x(),...
import numpy as np

boold = False  # messaggi debug

inputs = [1, 2, 3, 2.5]          # input
weights = [0.2, 0.8, -0.5, 1.0]  # pesi dei collegamenti
bias = 2                         # errore statistico

if __name__ == "__main__":
    # prodotto interno + errore statistico
    output = np.dot(weights, inputs) + bias

    if boold:
        print("Prodotto interno tra:")
        print("* Inputs: ", inputs)
        print("* Pesi: ", weights)
        print("* Errore statistico: ", bias)
        print("Output:")

    print(output)

    # -- SEZIONE TEST
    # calcola il risultato desiderato
    desired_output = utils.calc_neuron(inputs, weights, bias)

    if boold:
        print("Output desiderato:")
        print(desired_output)

    # errore se il risultato e' diverso da quello desiderato
    assert output == desired_output
