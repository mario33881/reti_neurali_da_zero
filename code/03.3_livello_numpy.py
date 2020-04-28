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
gli output di un layer (come nel video 2)
utilizzando la libreria numpy per calcolare
il prodotto interno (dot product).

Il prodotto interno calcola il prodotto
tra i singoli elementi di due liste dello stesso indice
> elemento 0 * elemento 0, elemento 1 * elemento 1,...
e somma i prodotti insieme.

Per ogni neurone:
* Viene usato il prodotto interno per calcolare
  la moltiplicazione tra input e pesi e sommare i risultati.
* Al prodotto interno viene sommato l'errore statistico (bias)

In uscita al layer otteniamo un array di 3 output (uno per neurone)

Documentazione di numpy.dot():
https://numpy.org/doc/stable/reference/generated/numpy.dot.html?highlight=dot#numpy.dot
"""
# possiamo richiamare oggetti (funzioni,...) numpy facendo np.x(),...
import numpy as np

# input
inputs = [1, 2, 3, 2.5]

# matrice dei pesi dei collegamenti
weights = [
    [
        0.2,  # peso collegamento input 1 - neurone 1
        0.8,  # peso collegamento input 2 - neurone 1
        -0.5, # peso collegamento input 3 - neurone 1
        1.0   # peso collegamento input 4 - neurone 1
    ],

    # pesi collegamenti verso il neurone 2
    [
        0.5,   # peso collegamento input 1 - neurone 2
        -0.91, # peso collegamento input 2 - neurone 2
        0.26,  # peso collegamento input 3 - neurone 2
        -0.5   # peso collegamento input 4 - neurone 2
    ],

    # pesi collegamenti verso il neurone 3
    [
        -0.26, # peso collegamento input 1 - neurone 3
        -0.27, # peso collegamento input 2 - neurone 3
        0.17,  # peso collegamento input 3 - neurone 3
        0.87   # peso collegamento input 4 - neurone 3
    ]
]

# errori statistici
biases = [2,   # errore statistico neurone 1
          3,   # errore statistico neurone 2
          0.5  # errore statistico neurone 3
]


if __name__ == "__main__":
        
    # prodotto interno
    output = np.dot(weights, inputs) + biases

    print(output)