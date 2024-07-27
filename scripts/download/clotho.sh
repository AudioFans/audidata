#!/bin/bash

# Dataset link: https://zenodo.org/records/3490684

mkdir clotho
cd clotho

# Download from Zenodo
wget -O clotho_audio_development.7z https://zenodo.org/records/3490684/files/clotho_audio_development.7z?download=1

wget -O clotho_audio_evaluation.7z https://zenodo.org/records/3490684/files/clotho_audio_evaluation.7z?download=1

wget -O clotho_captions_development.csv https://zenodo.org/records/3490684/files/clotho_captions_development.csv?download=1

wget -O clotho_captions_evaluation.csv https://zenodo.org/records/3490684/files/clotho_captions_evaluation.csv?download=1

wget -O clotho_metadata_development.csv https://zenodo.org/records/3490684/files/clotho_metadata_development.csv?download=1

wget -O clotho_metadata_evaluation.csv https://zenodo.org/records/3490684/files/clotho_metadata_evaluation.csv?download=1

wget -O LICENSE https://zenodo.org/records/3490684/files/LICENSE?download=1

# Decompress
7z e clotho_audio_development.7z -oclotho_audio_development

7z e clotho_audio_evaluation.7z -oclotho_audio_evaluation