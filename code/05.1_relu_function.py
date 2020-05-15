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

Questo script usa una rectified linear function (ReLU), in italiano
rettificatore, per ottenere un output di soli valori positivi.

"""

__author__ = "Zenaro Stefano"
__version__ = "01_01 2020-05-14"

import utils  # per funzioni di test

boold = False  # per debug

# valori in input alla ReLU
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

# output di ReLU
output = []      # output con condizioni
output_max = []  # output con max()

if __name__ == "__main__":

    # calcolo output con condizioni
    for i in inputs:
        if i > 0:
            output.append(i)
        elif i <= 0:
            output.append(0)

    # calcolo output con funzione max()
    for i in inputs:
        # se 0 e' il valore piu' grande, aggiungi 0
        # altrimenti aggiungi l'input <i>
        output_max.append(max(0, i))

    if boold:
        print("output max():")
        print(output_max)
        print("output condizioni:")

    print(output)

    # -- SEZIONE TEST
    # calcola il risultato aspettato
    # > inputs lo consideriamo come output dei singoli neuroni
    # > a cui dobbiamo ancora applicare la ReLU
    expected_output = utils.layer_relu_function(inputs)

    # l'output con la condizione dovrebbe dare il risultato aspettato
    assert(output == expected_output)

    # l'output con max() dovrebbe dare il risultato aspettato
    assert(output_max == expected_output)
