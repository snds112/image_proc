# Image Processing Studio

Application interactive de traitement d'image avec Streamlit et OpenCV. On upload une image, on empile une chaîne d'effets depuis la sidebar, et on compare le résultat à l'original.

## Contenu

- `app.py` — interface Streamlit
- `image_effects.py` — implémentation des effets
- `styles.py` — CSS personnalisé

## Effets disponibles

- **Couleur** : grayscale, inversion, luminosité, contraste, étirement et égalisation d'histogramme
- **Bruit** : sel & poivre, gaussien, périodique, poisson
- **Géométrique** : rotation, redimensionnement, miroir
- **Filtres** : seuillage, flou gaussien, netteté
- **Contours** : Canny, Roberts, Prewitt, Sobel, Laplacien, LoG
- **Morphologie** : dilation, érosion, ouverture, fermeture
- **Clustering** : K-Means, Fuzzy C-Means, Probabilistic C-Means

## Utilisation

```bash
pip install streamlit opencv-python numpy scipy matplotlib pillow
streamlit run app.py
```

## Fonctionnalités

- Chaîne d'effets empilables, activables/désactivables individuellement
- Comparaison original / traité côte à côte
- Calcul du MSE et du PSNR
- Histogrammes RGB ou niveaux de gris
- Téléchargement de l'image traitée
