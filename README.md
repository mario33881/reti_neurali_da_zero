# Reti neurali da zero

Appunti e traduzione della serie "Neural Networks from Scratch"
su Youtube di sentdex: [clicca qui per accedere al suo canale Youtube](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)

Per supportare l'autore considerare l'acquisto del
libro dal sito: https://nnfs.io/

Inoltre: guardare anche i suoi video per
usufruire di animazioni, ecc...

> Non sono associato a Sentdex, il suo canale, il suo sito e il suo libro

E' possibile seguire la serie conoscendo altri linguaggi di programmazione
andando sul seguente repository: [clicca qui](https://github.com/Sentdex/NNfSiX)

Notes and translation of the "Neural Networks from Scratch" series.
It's on Youtube and the author is sentdex: [click here to go to his Youtube channel](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)

To support the author consider buying the book
at this website: https://nnfs.io/

Also: Watch the videos because
they have animations, etc...

> I'm not associated to Sentdex, his channel, his website, his book in any way, shape or form

You can follow the series knowing other programming languages
by going to this repository: [click here](https://github.com/Sentdex/NNfSiX)

---

Obiettivo: capire come funzionano le reti neurali
dalle radici

Requisiti: 
* conoscenze: basi di programmazione e programmazione ad oggetti
* python 3 con numpy(libreria per operazioni matematiche) e matplotlib(visualizzazione grafici)

## Indice
* [Video 1: Intro e codice di un neurone](#video-1-intro-e-codice-di-un-neurone)
	* [Video](https://www.youtube.com/watch?v=Wo5dMEP_BbI)
    * [Codice](code/01_neurone.py)
* [Video 2: codificare un livello](#video-2-codificare-un-livello)
    * [Video](https://www.youtube.com/watch?v=lGLto9Xd7bU)
	* [Codice](code/02_livello.py)
* [Video 3: prodotto interno (dot product)](#video-3-prodotto-interno-dot-product)
    * [Video](https://www.youtube.com/watch?v=tMrbN67U9d4)
    * [Codice video 2 dinamico](code/03.1_livello_codice_dinamico.py)
    * [Codice video 1 numpy](code/03.2_neurone_numpy.py)
    * [Codice video 2 numpy](code/03.3_livello_numpy.py)
* [Changelog](#changelog)

> Nota 1: clicca sui titoli esterni per andare nelle sezioni

> Nota 2: i sottotitoli indentati ai titoli principali portano
a risorse aggiuntive (video, codice, ...)

## Video 1: Intro e codice di un neurone
> Link: https://www.youtube.com/watch?v=Wo5dMEP_BbI

### Introduzione
Una rete neurale ha dei dati in input.

Ogni dato (proveniente da un input o da un neurone) e' legato
ad altri neuroni: ogni legame ha il proprio peso ("weight").

Il dato viene moltiplicato per il peso e sommato
a tutti gli altri risultati di moltiplicazione dato-peso.

Infine questo dato viene sommato ad un errore statistico("bias") e
viene passato ad una funzione di attivazione ("activation function").
> l'attivazione e' la simulazione dell'impulso di un neurone naturale

![](assets/svg/01-neurone_generico.svg)

Questo passaggio viene ripetuto per tutti gli strati (i "layer").

Dal risultato viene ottenuta la perdita/errore ("loss"), cioe' quanto sbaglia
la rete neurale: questa informazione ci permette di migliorare la rete neurale.


Tutti i calcoli matematici sono fonte di logaritmi, esponenziali, somme, ...
calcoli infinitesimali e algebra lineare.

In una rete neurale:
* sono presenti i neuroni, connessi insieme
* strato di input ("input layer")
* strati nascosti ("hidden layer")
* strato di output ("output layer")

L'obiettivo e' prevedere attraverso un nuovo input quale e' l'output desiderato.
Per farlo viene utilizzato il processo di apprendimento ("training process") in cui
vengono modificati i pesi dei collegamenti e gli errori statistici.

La rete neurale e' quindi formato da neuroni di piu' strati tutti
collegati fra loro e ogni collegamento, grazie ai pesi, permette
di avere diversi parametri e relazioni.

La parte piu' difficile da capire di una rete neurale e' come
modificare i pesi dei singoli collegamenti.

### Codice di un neurone

![](assets/svg/01-neurone_calcoli.svg)

Consideriamo 3 neuroni che mandano il proprio
output al nostro neurone:
```python
inputs = [1.2, # output neurone 1
          5.1, # output neurone 2
          2.1  # output neurone 3
]
```

Se abbiamo 3 neuroni che passano dati
al nostro neurone abbiamo 3 collegamenti,
ognuno con il proprio peso:
```python
weights = [3.1, # peso collegamento neurone 1 - nostro neurone
           2.1, # peso collegamento neurone 2 - nostro neurone
           8.7  # peso collegamento neurone 3 - nostro neurone
]
```

Ogni neurone, quindi anche il nostro,
ha un errore statistico:
```python
bias = 3 # errore statistico
```

Il compito del neurone e' moltiplicare i dati
per il peso del collegamento, sommare i risultati
fra di loro e con l'errore statistico:
```python
output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
```
Questo e' l'output del neurone: 35.7

Codice completo:
```python
inputs = [1.2, # output neurone 1
          5.1, # output neurone 2
          2.1  # output neurone 3
]

weights = [3.1, # peso collegamento neurone 1 - nostro neurone
           2.1, # peso collegamento neurone 2 - nostro neurone
           8.7  # peso collegamento neurone 3 - nostro neurone
]

bias = 3 # errore statistico

output = (inputs[0] * weights[0] +  # dato neurone 1 moltiplicato per peso collegamento 
          inputs[1] * weights[1] +  # dato neurone 2 moltiplicato per peso collegamento
          inputs[2] * weights[2] +  # dato neurone 3 moltiplicato per peso collegamento
          bias)                     # tutte moltiplicazioni sommate insieme con l'errore statistico.
```
> Il codice completo e' disponibile al percorso: [```code/01_neurone.py```](code/01_neurone.py)


[Torna all'indice](#indice)

---

## Video 2: codificare un livello
> Link: https://www.youtube.com/watch?v=lGLto9Xd7bU

![](assets/svg/02-layer_calcoli.svg)

Il video inizia partendo dal codice dell'altra volta (con valori modificati):
> Nota: i numeri sono inventati quindi il fatto che siano
        diversi non implica ragionamenti diversi sul funzionamento del neurone

```python
inputs = [1, # output neurone 1
          2, # output neurone 2
          3  # output neurone 3
]

weights = [0.2,  # peso collegamento neurone 1 - nostro neurone
           0.8,  # peso collegamento neurone 2 - nostro neurone
           -0.5  # peso collegamento neurone 3 - nostro neurone
]

bias = 2 # errore statistico

output = (inputs[0] * weights[0] +  # dato neurone 1 moltiplicato per peso collegamento 
          inputs[1] * weights[1] +  # dato neurone 2 moltiplicato per peso collegamento
          inputs[2] * weights[2] +  # dato neurone 3 moltiplicato per peso collegamento
          bias)                     # tutte moltiplicazioni sommate insieme con l'errore statistico.
```

Gli input del neurone possono provvenire dall'input layer (valori "presi dalla realta'")
o da un layer nascosto (valori in output da altri neuroni).

Nel momento in cui aumenta il numero di input del neurone aumenta
della stessa quantita' il numero di pesi (es. 4 input, 4 pesi), 
mentre l'errore statistico e' 1 solo per neurone.


Considerando 3 neuroni con quattro input, ogni neurone ha:
* 4 input
* 4 pesi (1 per input)
* 1 bias (errore statistico)

Dove l'input per ogni neurone e':
```python
inputs = [1,  # input 1
          2,  # input 2
          3,  # input 3
          2.5 # input 4
]
```

In codice:
* i "set di pesi" sono 3:
    ```python
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
    ```
* i bias sono 3:
    ```python
    bias1 = 2 # errore statistico neurone 1

    bias2 = 3 # errore statistico neurone 2

    bias3 = 0.5 # errore statistico neurone 3
    ```

Se abbiamo 3 neuroni, e ogni neurone ha 1 output, otterremo un set di 3 output:
```python
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
```

Output dei neuroni: ```[4.8, 1.21, 2.385]```

Se vogliamo ottenere l'output desiderato la rete neurale:
* NON puo' modificare gli input (sono risultato di calcoli effettuati da altri neuroni o i dati di partenza provenienti ad esempio da sensori, ecc...)
* e' possibile modificare i pesi (weights) e gli errori statistici (bias). Il Deep learning modifica questi valori

Questa parte verra' trattata nei prossimi video

> Il codice completo e' disponibile al percorso: [```code/02_livello.py```](code/02_livello.py)


[Torna all'indice](#indice)

---

## Video 3: prodotto interno (dot product)
> Link: https://www.youtube.com/watch?v=tMrbN67U9d4

Obiettivo: usare liste e matrici nel codice
per renderlo piu' dinamico.

Creare una matrice di pesi:
```python
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
```

E una lista di errori statistici:
```python
biases = [2,   # errore statistico neurone 1
          3,   # errore statistico neurone 2
          0.5  # errore statistico neurone 3
]
```

Per il resto del codice:
```python
# output del layer (lista di output dei neuroni)
layer_outputs = []

# scorri unione delle liste dei pesi e dei bias
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
```

> Il codice completo e' disponibile al percorso: [```code/03.1_livello_codice_dinamico.py```](code/03.1_livello_codice_dinamico.py)


Di cosa si occupano i pesi e gli errori statistici:
* i pesi amplificano (/attenuano) un certo input
* l'errore statistico compensa il risultato (esegue un offset)

Questi pesi e errori statistici permettono di ottenere l'output
desiderato.

### Il concetto di "Shape"
E' la dimensione di una lista, lista di liste,...

Esempio:
una lista di quattro elementi:
```python
l = [1, 5, 6, 2]
```
Ha come "Shape": ```(4)```

perche' e' una lista (di una dimensione) di 4 elementi


Se due liste sono contenute in una lista (matrice, array di due dimensioni):
```python
lol = [
    [1, 5, 6, 2],
    [3, 2, 1, 3]
]
```
Ha come "Shape": ```(2, 4)```

perche' e' una matrice di due dimensioni (contiene due liste)
e che a loro volta hanno 4 elementi.

Attenzione: le liste devono essere omologhe,
non possono esserci liste di dimensioni diverse
come ad esempio:
```python
lol = [
    [1, 5, 6, 2],
    [3, 2, 1]
]
```


Se ho 3 liste che contengono ciascuna due liste
di elementi (array di 3 dimensioni):
```python
lolol = [
    
    [
        [1, 5, 6, 2],
        [3, 2, 1, 3]
    ],

    [
        [5,2,1,2],
        [6,4,8,4]
    ],

    [
        [2,8,5,3],
        [1,1,9,4]
    ]
    
]
```
Ha una "Shape": ```(3, 2, 4)```.

Un tensore ("tensor"), usato nel deep learning, puo' essere rappresentato
come un array di array

### Prodotto interno - dot product
In matematica l'operazione di prodotto
interno consiste nel prendere in
ordine gli elementi di due vettori,
di moltiplicare gli elementi e infine
di sommare i risultati.

Il risultato e' un valore.

Esempio:
```python

a= [1, 2, 3]
b = [2, 3, 4]

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
#              1  * 2   +  2 * 3    +  3 *  4
#                 2     +    6      +    12
# 
# dot_product = 20
```


Per eseguire il prodotto interno
di un neurone utilizzando la libreria numpy:
```python
# possiamo richiamare oggetti (funzioni,...) numpy facendo np.x(),...
import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

# prodotto interno
output = np.dot(weights, inputs) + bias
```
> nota: in questo caso, poiche' input e pesi sono entrambi
> vettori, e' possibile invertire gli argomenti della funzione
> ma e' meglio passare prima i pesi e poi gli input

> Il codice completo e' disponibile al percorso: [```code/03.2_neurone_numpy.py```](code/03.2_neurone_numpy.py)

Per eseguire il prodotto interno
di un layer utilizzando numpy:
```python
# possiamo richiamare oggetti (funzioni,...) numpy facendo np.x(),...
import numpy as np

inputs = [1, 2, 3, 2.5]
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


biases = [2,   # errore statistico neurone 1
          3,   # errore statistico neurone 2
          0.5  # errore statistico neurone 3
]

# prodotto interno
output = np.dot(weights, inputs) + biases
```
> Il codice completo e' disponibile al percorso: [```code/03.3_livello_numpy.py```](code/03.3_livello_numpy.py)

In questo caso è necessario passare prima i pesi
e poi gli input perche' numpy fornisce un output
in base alla struttura del primo parametro passato:
passando una matrice con 3 array indico che ci sono
3 risultati, uno per neurone.

Internamente numpy calcola i prodotti interni
tra i vettori nella matrice e il vettore degli input:
```python
np.dot(weights, inputs) = [
    np.dot(weights[0], inputs), # prodotto interno tra pesi neurone 1 e input
    np.dot(weights[1], inputs), # prodotto interno tra pesi neurone 2 e input
    np.dot(weights[2], inputs)  # prodotto interno tra pesi neurone 3 e input
]
```

Dopo aver calcolato il vettore dei prodotti interni
vengono sommati gli errori statistici:
prodotto interno 0 con errore statistico 0,
prodotto interno 1 con errore stat. 1,
prodotto interno 2 con errore stat. 2

[Torna all'indice](#indice)

## Changelog

**Commit 3 2020-04-28:** <br>
* aggiunti appunti del terzo video
* aggiunti i link ai video anche nell'indice
* aggiunto link al repository di github 
con codici in altri linguaggi di programmazione: [link](https://github.com/Sentdex/NNfSiX)
* aggiunti ```<hr>``` di separazione tra i video
* aggiunto script e dizionari per automatizzare il controllo
di errori di scrittura con hunspell

**Commit 2 2020-04-19:** <br>
* Sostituiti i tab in spazi (README e codice)
* Sistemati i link nell'indice
* Aggiunte immagini relative ai primi due video

**Commit 1 2020-04-19:** <br>
Primo commit: appunti dei primi 2 video

## Autore
* original author: Sentdex ([Youtube channel](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ))
* notes and translation: mario33881