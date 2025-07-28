#!/bin/bash

# Créer un dossier pour les fichiers téléchargés
mkdir -p cityscapes_data
cd cityscapes_data

# URLs à télécharger
URL1="https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_gtFine_trainvaltest.zip"
URL2="https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_leftImg8bit_trainvaltest.zip"

# Télécharger les fichiers avec curl
echo "Téléchargement de P8_Cityscapes_gtFine_trainvaltest.zip..."
curl -O "$URL1"

echo "Téléchargement de P8_Cityscapes_leftImg8bit_trainvaltest.zip..."
curl -O "$URL2"

# Vérifier si les fichiers sont bien téléchargés
if [[ -f "P8_Cityscapes_gtFine_trainvaltest.zip" && -f "P8_Cityscapes_leftImg8bit_trainvaltest.zip" ]]; then
    echo "Téléchargement terminé avec succès."
else
    echo "Erreur lors du téléchargement d'un ou plusieurs fichiers."
fi

