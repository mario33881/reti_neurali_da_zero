"""
# Video 1: codice neurone

Appunti e traduzione della serie "Neural Networks from Scratch"
su Youtube di sentdex: [clicca qui per accedere al suo canale Youtube](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)

Per supportare l'autore considerare l'acquisto del
libro dal sito: https://nnfs.io/

> Non sono associato a Sentdex, il suo canale, il suo sito e il suo libro

Notes and translation of the "Neural Networks from Scratch" series.
It's on Youtube and the author is sentdex: [click here to go to his Youtube channel](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)

To support the author consider buying the book
at this website: https://nnfs.io/

> I'm not associated to Sentdex, his channel, his website, his book in any way, shape or form

---

Il primo video codifica un neurone:
* il neurone ha degli input (provenienti da altri neuroni o da dati reali)
> dati reali = dati provenienti da sensori, ecc...
* gli input sono connessi al neurone e ogni collegamento ha un peso
* e' presente un errore statistico

Il neurone:
* riceve gli input moltiplicati per il peso del collegamento
* somma i prodotti delle moltiplicazioni
* somma l'errore statistico

"""

__author__ = "Zenaro Stefano"
__version__ = "01_01 2020-04-19"

boold = False

inputs = [1.2, # output neurone 1
          5.1, # output neurone 2
		  2.1  # output neurone 3
]

weights = [3.1, # peso collegamento neurone 1 - nostro neurone
           2.1, # peso collegamento neurone 2 - nostro neurone
		   8.7  # peso collegamento neurone 3 - nostro neurone
]

bias = 3 # errore statistico

if __name__ == "__main__":

    if boold:
        print("Inizio programma")

    output = (inputs[0] * weights[0] +  # dato neurone 1 moltiplicato per peso collegamento 
              inputs[1] * weights[1] +  # dato neurone 2 moltiplicato per peso collegamento
		      inputs[2] * weights[2] +  # dato neurone 3 moltiplicato per peso collegamento
		      bias)                     # tutte moltiplicazioni sommate insieme con l'errore statistico.

    print(output)
	
    if boold:
        print("Fine programma")
