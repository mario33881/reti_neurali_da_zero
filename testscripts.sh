#!/bin/bash

#
# TESTSCRIPTS: esegue script nella cartella code
#              per garantire il loro funzionamento
#
# lo script usa il comando "find" per trovare i file python nella cartella code
# e ogni file trovato viene eseguito con python
#
# Autore: Zenaro Stefano
# Versione: 01_01 2020-05-04
#

# esci al primo errore e con il codice di uscita dell'ultimo comando
set -e
set -o pipefail

# percorso di questo script
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# percorso della cartella code
CODEPATH="${SCRIPTPATH}/code"

# trova in code, file di tipo "file", nome qualsiasi ("*") che finisce con estensione ".py":
# per ogni file esegui "python <file>"
find "$CODEPATH" -type f -iname "*.py" -exec python {} \;