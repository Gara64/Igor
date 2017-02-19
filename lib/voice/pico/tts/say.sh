#!/bin/bash
if [ "$1" == "" ]
then
  MSG="paràmetro mancante: frase da pronunciare."
else
  MSG=${1}
fi
export LD_LIBRARY_PATH=. ; ./picoPi2Std "${MSG}" | aplay -f S16_LE -r 16000
