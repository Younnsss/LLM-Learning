# TP2 — Fine-tuning GPT (Spam Detection)

- **Nom / Prénom** : Boutkrida Younes
- **OS** : Ubnutu
- **Python** : Python 3.12.3
- **Seed fixé** : 42

## `settings`

`settings` est un dictionnaire (`dict`) chargé à partir du fichier `hparams.json`.
Justification (extrait de notebook) : type, liste des clés, et valeurs des champs principaux (`n_vocab`, `n_ctx`, `n_embd`, `n_head`, `n_layer`).

Number of keys: 5
Keys: ['n_vocab', 'n_ctx', 'n_embd', 'n_head', 'n_layer']
n_vocab = 50257
n_ctx = 1024
n_embd = 768
n_head = 12
n_layer = 12

## `params`

`params` est un dictionnaire (`dict`) contenant les poids du modèle GPT-2 extraits du checkpoint TensorFlow.  
Les éléments notables incluent : `wte`, `wpe`, `g`, `b`, ainsi qu'une liste `blocks` contenant `n_layer` éléments (un bloc par couche Transformer).

Type(params): <class 'dict'>
Keys(params): ['blocks', 'b', 'g', 'wpe', 'wte']
Type(params['blocks']): <class 'list'>
len(params['blocks']): 12
Example block[0] keys: ['attn', 'ln_1', 'ln_2', 'mlp']
wte shape: (50257, 768)
wpe shape: (1024, 768)
g shape: (768,)
b shape: (768,)

## 5.1

Cette ligne de code effectue un mélange des données du dataframe tout en utilisant une seed fixée, garantissant ainsi la reproductibilité du mélange.

## 5.2

Label distribution: ham - 86.61%, spam - 13.39%
Le dataset présente un déséquilibre significatif, avec une proportion nettement inférieure de données liées aux spams.

## 8.3

Les paramètres sont gelés avec cette ligne de code afin de préserver la capacité de généralisation du modèle original tout en adaptant celui-ci à notre tâche spécifique par le fine-tuning.

### 10

Au cours de l'entraînement, on observe une diminution initiale de la loss, suivie d'une stagnation rapide. La précision reste élevée (84%) durant les deux premières époques, mais le taux de détection des spams demeure à 0%. Lors de la dernière époque, le modèle adopte un comportement radicalement opposé en prédisant uniquement des spams, ce qui entraîne une précision globale de 16%. Ce comportement illustre que le modèle n'a pas appris correctement : il bascule entre prédire uniquement les hams ou uniquement les spams, sans véritable généralisation.

Répartition des labels :
ham : 86.61%
spam : 13.39%

La précision mesurée reflète directement cette répartition initiale, confirmant l'absence d'apprentissage significatif par le modèle.
