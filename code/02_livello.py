"""
# Video 2: codificare livello

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

Il secondo video codifica un livello, il livello ha:
* 4 input per neurone (e di conseguenza 4 pesi di collegamento, 1 per input)
* 3 neuroni (e di conseguenza 3 bias e 3 output)

Ogni neurone ha quindi:
* 4 input
* 4 collegamenti ai neuroni del layer precedente (1 per input)
* 1 bias (errore statistico)
* 1 output

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

# questi sono gli input dei 3 neuroni
inputs = [1,  # input 1
		  2,  # input 2
		  3,  # input 3
		  2.5 # input 4
]

# -- PESI COLLEGAMENTI

# pesi collegamenti verso il neurone 1
weights1 = [0.2,  # peso collegamento input 1 - neurone 1
		    0.8,  # peso collegamento input 2 - neurone 1
		    -0.5, # peso collegamento input 3 - neurone 1
		    1.0   # peso collegamento input 4 - neurone 1
]

# pesi collegamenti verso il neurone 2
weights2 = [0.5,   # peso collegamento input 1 - neurone 2
		    -0.91, # peso collegamento input 2 - neurone 2
		    0.26,  # peso collegamento input 3 - neurone 2
		    -0.5   # peso collegamento input 4 - neurone 2
]

# pesi collegamenti verso il neurone 3
weights3 = [-0.26, # peso collegamento input 1 - neurone 3
		    -0.27, # peso collegamento input 2 - neurone 3
		    0.17,  # peso collegamento input 3 - neurone 3
		    0.87   # peso collegamento input 4 - neurone 3
] 

# -- ERRORI STATISTICI

bias1 = 2 # errore statistico neurone 1

bias2 = 3 # errore statistico neurone 2

bias3 = 0.5 # errore statistico neurone 3


if __name__ == "__main__":

    if boold:
        print("Inizio programma")

    output = [  # CALCOLO OUTPUT NEURONE 1
	           (inputs[0] * weights1[0] + # dato input 1 moltiplicato per peso collegamento al neurone 1 
                inputs[1] * weights1[1] + # dato input 2 moltiplicato per peso collegamento al neurone 1
		        inputs[2] * weights1[2] + # dato input 3 moltiplicato per peso collegamento al neurone 1
				inputs[3] * weights1[3] + # dato input 4 moltiplicato per peso collegamento al neurone 1
		        bias1),                   # tutte moltiplicazioni sommate insieme con l'errore statistico del neurone 1.
		        
		        # CALCOLO OUTPUT NEURONE 2
		       (inputs[0] * weights2[0] + # dato input 1 moltiplicato per peso collegamento al neurone 2 
                inputs[1] * weights2[1] + # dato input 2 moltiplicato per peso collegamento al neurone 2
		        inputs[2] * weights2[2] + # dato input 3 moltiplicato per peso collegamento al neurone 2
				inputs[3] * weights2[3] + # dato input 4 moltiplicato per peso collegamento al neurone 2
		        bias2),                 + # tutte moltiplicazioni sommate insieme con l'errore statistico del neurone 2.
		        
		        # CALCOLO OUTPUT NEURONE 3
		       (inputs[0] * weights3[0] + # dato input 1 moltiplicato per peso collegamento al neurone 3 
                inputs[1] * weights3[1] + # dato input 2 moltiplicato per peso collegamento al neurone 3
		        inputs[2] * weights3[2] + # dato input 3 moltiplicato per peso collegamento al neurone 3
				inputs[3] * weights3[3] + # dato input 4 moltiplicato per peso collegamento al neurone 3
		        bias3),                   # tutte moltiplicazioni sommate insieme con l'errore statistico del neurone 3.
    ]

    print(output)
	
    if boold:
        print("Fine programma")
