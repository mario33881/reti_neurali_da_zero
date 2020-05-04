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

Per rendere il codice del video 2 piu'
dinamico e' possibile:
* creare una matrice (lista di liste)
con i pesi dei collegamenti
* creare una lista di errori statistici
* collezionare l'output in una lista

Questo codice verra' modificato nel video 3 stesso
per implementare la libreria numpy.
"""

__author__ = "Zenaro Stefano"
__version__ = "01_01 2020-04-28"

import utils  # per funzioni di test

boold = False  # messaggi di debug

# questi sono gli input dei 3 neuroni
inputs = [1,  # input 1
          2,  # input 2
          3,  # input 3
          2.5  # input 4
          ]

# lista di liste con i pesi
weights = [
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

# lista con pesi statistici
biases = [2,   # errore statistico neurone 1
          3,   # errore statistico neurone 2
          0.5  # errore statistico neurone 3
          ]


# output del layer (lista di output dei neuroni)
layer_outputs = []

if __name__ == "__main__":

    # scorri unione (fatta con zip() ) delle liste dei pesi e dei bias
    #
    # Documentazione di zip(): https://docs.python.org/3/library/functions.html#zip
    # 
    # > l'unione e' fatta per elemento 
    # > (es. elemento 0 di weights e elemento 0 di biases sono il primo array,
    # >      elemento 1 di weights e elemento 1 di biases sono il secondo array, ...)
    for neuron_weights, neuron_bias in zip(weights, biases):
        # output nel neurone
        neuron_output = 0
        
        # scorri unione degli input e dei pesi
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight  # moltiplica (input * peso) per input e sommali
        
        # somma l'errore statistico
        neuron_output += neuron_bias

        # aggiungi il risultato alla lista degli output del layer
        layer_outputs.append(neuron_output)

        if boold:
            print("Il neurone '{}' ha:".format(len(layer_outputs)))
            print("* Input: ", inputs)
            print("* Pesi: ", neuron_weights)
            print("* Errore statistico: ", neuron_bias)
            print("Output del neurone: ", neuron_output)
            print("-" * 50)
    
    if boold:
        print("Output del layer:")
        
    print(layer_outputs)

    # -- SEZIONE DI TEST

    # memorizza l'output desiderato dei neuroni
    desired_output = utils.calc_layer_by_props(inputs, weights, biases)

    if boold:
        print("Output desiderato:")
        print(desired_output)
        
    # errore se il risultato e' diverso da quello desiderato
    assert layer_outputs == desired_output
