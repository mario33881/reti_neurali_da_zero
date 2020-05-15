#!/bin/bash

#
# SPELLCHECK: controlla se sono presenti errori
#             di scrittura nel README
#
# lo script usa hunspell con il dizionario inglese,
# italiano e un dizionario personalizzato per assicurarsi
# che le parole siano valide e esistenti
#
# Autore: Zenaro Stefano
# Versione: 01_02 2020-04-28
#

# esci al primo errore e con il codice di uscita dell'ultimo comando
set -e
set -o pipefail

# percorso di questo script
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# percorso del file readme
READMEPATH="${SCRIPTPATH}/README.md"

# vai nella cartella con i dizionari di hunspell
cd "${SCRIPTPATH}/assets/hunspell"

# togli duplicati dalla wordlist e riordinala. Scrivi il risultato in file temporaneo
sort -u custom_dict-wordlist.txt | sort > custom_dict-wordlist.temp
# sovrascrivi la wordlist rinominando il file temporaneo
mv custom_dict-wordlist.temp custom_dict-wordlist.txt

# sovrascrivi il file dizionario con il numero di parole nella wordlist
wc -l custom_dict-wordlist.txt > custom_dict.dic
# riordina le parole nella word list, togli duplicati e aggiungile a fine dizionario
sort custom_dict-wordlist.txt | uniq >> custom_dict.dic

# memorizza output
output=$(hunspell -d en-GB,it_IT,custom_dict -l < "${READMEPATH}" -u)

# visualizza output
echo "$output"

# verifica se l'output e' vuoto (0 = vuoto, 1 = non vuoto)
[ -z "$output" ]