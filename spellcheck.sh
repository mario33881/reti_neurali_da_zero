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
# Versione: 01_01 2020-04-28
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

# memorizza output
output=$(hunspell -d en-GB,it_IT,custom_dict -l < "${READMEPATH}" -u)

# visualizza output
echo "$output"

# verifica se l'output e' vuoto (0 = vuoto, 1 = non vuoto)
[ -z "$output" ]