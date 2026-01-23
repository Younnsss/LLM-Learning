# TP3

```text
Lauret Alexandre
Commande d'installation : mamba env create -f csc8614/TP1/requirements.txt -n csc8614
Version Python/Librairie : Python 3.12.3
torch 2.9.1
transformers 4.57.3
plotly 6.5.1
sklearn 1.8.0

Seed utilisée : 42
```

## Question 1

![alt text](image.png)

On peut voir que chacune des couches linéaires est maintenant adaptée avec lora (LinearWithLora)

## Question 2

trainable params: 1,735,304 || all params: 164,772,488 || trainable%: 1.05%
On voit qu'il y a plus de paramètres globaux mais nous entraînons seulement une petite partie des paramètres afin de ne pas porter atteinte à la généralisation et l'entraînement obtenu en amont.

## Question 3

trainable params: 1,328,642 || all params: 125,768,450 || trainable%: 1.06%
On réduit le nombre total de paramètre car nous remplacons la tête de classification. On remplace la couche avec lora par une simple couche linéaire. En ayant retiré des paramètres entraînables et non entraînables, le % reste presque similaire.

## Question 4

Au fil des batchs, la loss diminue fortement avant de remonter. La précision finale est de 94% ce qui est un très bon score pour notre système de détection de spam. En seulement une époque nous atteignons une meilleure performance qu'au premier TP avec 3 époques.

## Question 5

L'accuracy obtenue sur le dataset de test est légèrement plus élevé que sur le dataset de train. Notre modèle est réellement performant et généraliste.
