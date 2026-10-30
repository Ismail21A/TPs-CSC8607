# TP4 — Réseaux récurrents et génération de séquences

## Exercice 1 — Exploration et prétraitement des partitions ABC

### Étape 1 — Chargement et découverte du jeu de données

Les fichiers JSON ont été chargés à l'aide de `json.load()`. Chaque entrée contient une partition musicale représentée sous forme textuelle, accessible par la clé `"abc notation"`.

Le jeu de données comprend :
- **214 122 partitions** pour l'entraînement ;
- **2 162 partitions** pour la validation.

La première partition du jeu d'entraînement est :

```abc
X:1
L:1/8
M:4/4
K:Emin
|: E2 EF E2 EF | DEFG AFDF | E2 EF E2 B2 |1 efe^d e2 e2 :|2 efe^d e3 B |: e2 ef g2 fe |
 defg afdf |1 e2 ef g2 fe | efe^d e3 B :|2 g2 bg f2 af | efe^d e2 e2 ||
```

**Interprétation des métadonnées :**

- `X:1` : identifiant de la partition.
- `L:1/8` : durée de référence des notes, correspondant à une croche.
- `M:4/4` : signature rythmique indiquant quatre temps par mesure.
- `K:Emin` : tonalité de mi mineur.

Le corps de la partition utilise des lettres pour représenter les notes, des nombres pour modifier leur durée et des symboles pour indiquer les mesures, les reprises et les altérations.

### Étape 2 — Construction du vocabulaire et padding

**1. Statistiques du jeu de données**

L'analyse des partitions donne les résultats suivants :

- Taille du vocabulaire : **95 caractères uniques**.
- Longueur moyenne : **290,1 caractères**.
- Longueur maximale : **2 968 caractères**.

Les partitions sont normalisées à une longueur fixe de 500 caractères par padding ou troncature.

**2. Pourquoi utiliser `sorted()` sur les caractères uniques ?**

Un `set` ne garantit pas un ordre stable entre les exécutions. L'utilisation de `sorted()` permet donc de construire un dictionnaire `char_to_index` déterministe.

Sans ce tri, les indices associés aux caractères pourraient changer entre deux exécutions. Si l'on rechargeait les poids d'un modèle sauvegardé avec un vocabulaire différent, les indices ne correspondraient plus aux mêmes caractères. Les embeddings et les prédictions seraient alors incorrects, même si les poids du modèle étaient identiques.

**3. Pourquoi choisir `SEQ_LENGTH = 500` plutôt que `max_len = 2968` ?**

La longueur moyenne des partitions est de 290,1 caractères. Une longueur de 500 permet donc de conserver intégralement de nombreuses partitions tout en limitant le padding et le coût de calcul.

Un padding jusqu'à 2 968 caractères produirait des séquences presque six fois plus longues, augmentant fortement le temps d'entraînement et la mémoire GPU nécessaire, notamment pour stocker les activations utilisées lors de la rétropropagation à travers le temps (BPTT).

De plus, une grande partie des positions correspondrait à des espaces ajoutés artificiellement. Le modèle risquerait alors d'apprendre principalement à prédire ces espaces plutôt que les structures musicales.

Le choix de 500 caractères constitue donc un compromis entre conservation de l'information musicale, efficacité de calcul et qualité de l'apprentissage.

## Exercice 2 — Dataset PyTorch auto-régressif et modélisation

### Étape 1 — Création du MusicDataset

Le `MusicDataset` transforme les partitions ABC en séquences d'indices numériques et prépare les données pour un apprentissage auto-régressif.

Pour chaque séquence de 500 caractères, l'entrée `x` contient les 499 premiers caractères (`seq[:-1]`), tandis que la cible `y` contient les 499 derniers (`seq[1:]`). Le modèle apprend ainsi à prédire le caractère suivant à chaque pas de temps.

Avec `SEQ_LENGTH = 500` et `BATCH_SIZE = 128`, les dimensions obtenues sont :

- **x :** `torch.Size([128, 499])`
- **y :** `torch.Size([128, 499])`

Le type `torch.long` (`int64`) est nécessaire car les caractères sont représentés par des indices entiers utilisés par la couche `nn.Embedding`. Le type `torch.float32` ne convient pas pour indexer directement cette couche.

Les tests réalisés sur un sous-ensemble de 10 000 partitions confirment les dimensions attendues et le décalage temporel correct entre les entrées et les cibles.

### Étape 2 — Architecture récurrente : RNN vs LSTM

**1. Calcul théorique du nombre de paramètres**

On considère un vocabulaire de taille \(V=95\), une dimension d'embedding \(E=256\) et une dimension cachée \(H=512\).

**Couche Embedding (identique pour les deux architectures)**

\[
P_{\text{embedding}} = V \times E = 95 \times 256 = 24\,320
\]

**RNN classique (Elman)**

La couche `nn.RNN` possède une matrice entrée-état, une matrice état-état et deux vecteurs de biais :

\[
P_{\text{RNN}} = HE + H^2 + 2H
\]

\[
= 512 \times 256 + 512^2 + 2 \times 512
= 394\,240
\]

**LSTM**

Le LSTM possède quatre ensembles de paramètres correspondant à la porte d'entrée, la porte d'oubli, la porte de sortie et la mise à jour candidate de la mémoire :

\[
P_{\text{LSTM}} = 4(HE + H^2 + 2H)
\]

\[
=4 \times 394\,240 = 1\,576\,960
\]

Le bloc LSTM possède donc exactement quatre fois plus de paramètres que le bloc RNN classique, car chaque porte et la mise à jour candidate disposent de leurs propres poids et biais.

**Couche linéaire de sortie**

\[
P_{\text{Linear}} = HV + V
= 512 \times 95 + 95 = 48\,735
\]

**Nombre total de paramètres :**

| Couche | RNN | LSTM |
|---|---:|---:|
| Embedding | 24 320 | 24 320 |
| Récurrente | 394 240 | 1 576 960 |
| Linear | 48 735 | 48 735 |
| **Total** | **467 295** | **1 650 015** |

Les valeurs calculées ont été vérifiées expérimentalement avec `count_parameters()` et correspondent exactement aux résultats obtenus avec PyTorch.

**2. Pourquoi utiliser `nn.Embedding` plutôt qu'une projection linéaire d'un indice scalaire ?**

Les indices des caractères sont des identifiants arbitraires et non des valeurs numériques représentant une grandeur.

Une transformation linéaire appliquée directement à un indice scalaire imposerait une relation artificielle entre les caractères : deux indices proches seraient traités comme des valeurs proches, et les représentations seraient contraintes à une même direction affine.

Au contraire, `nn.Embedding` apprend un vecteur indépendant pour chaque caractère. Cela permet au modèle de construire des représentations adaptées aux relations statistiques entre les caractères et aux structures musicales.

**3. Différence entre les états cachés RNN et LSTM**

Pour un RNN classique, `hidden_state` est un unique tenseur représentant l'état caché final :

`(num_layers, batch_size, hidden_size)`

Dans notre cas : `(1, 2, 512)`.

Pour un LSTM, `hidden_state` est un tuple `(h_n, c_n)` contenant deux tenseurs :

- `h_n` : état caché, utilisé pour produire les sorties et transmettre l'information au pas suivant.
- `c_n` : état de cellule, représentant la mémoire interne qui permet de conserver des informations sur des dépendances temporelles plus longues.

Les deux tenseurs ont ici la forme `(1, 2, 512)`.

Les tests confirment également que les deux modèles produisent des logits de forme `(2, 20, 95)` pour une entrée de deux séquences de vingt caractères.

## Exercice 3 — Entraînement et comparaison des modèles

### Protocole expérimental

Nous avons entraîné trois modèles de prédiction du caractère suivant sur des partitions musicales au format ABC : un RNN sans gradient clipping, un RNN avec gradient clipping et un LSTM avec gradient clipping.

Les expériences utilisent un sous-ensemble de 10 000 partitions pour l'entraînement, 2 162 partitions pour la validation, une longueur de séquence de 500 caractères, un batch size de 128, une dimension d'embedding de 256 et une taille d'état caché de 512. L'optimiseur utilisé est Adam avec un taux d'apprentissage de 0,001. Chaque modèle est entraîné pendant 15 époques, avec une patience de 5 époques pour l'early stopping.

Le gradient clipping est réalisé par `clip_grad_norm_` avec une norme maximale de 1,0. Les entraînements ont été exécutés sur GPU via Slurm. Les métriques ont été enregistrées avec TensorBoard et le meilleur modèle de chaque expérience a été sauvegardé selon la perte de validation.

### Résultats

| Modèle | Clipping | Meilleure perte de validation | Accuracy de validation | Époque optimale |
|---|---|---:|---:|---:|
| RNN | Non | 0,7041 | 77,73 % | 14 |
| RNN | Oui (1,0) | 0,6791 | 78,45 % | 15 |
| LSTM | Oui (1,0) | **0,5658** | **81,59 %** | 14 |

### Analyse

Les trois modèles présentent une diminution globale de la perte d'entraînement et de validation, ce qui confirme le bon fonctionnement du processus d'apprentissage.

Le RNN avec gradient clipping obtient une perte de validation légèrement inférieure à celle du RNN sans clipping. Cependant, les normes moyennes des gradients restent généralement inférieures au seuil de 1,0. Cette expérience ne permet donc pas, à elle seule, de conclure que le clipping est responsable de l'amélioration observée.

Le LSTM obtient les meilleurs résultats, avec une perte de validation de 0,5658 contre 0,6791 pour le RNN avec clipping. Cette amélioration est cohérente avec la capacité du LSTM à gérer les dépendances temporelles grâce à ses mécanismes de portes et à son état mémoire.

L'accuracy calculée inclut les caractères de remplissage (*padding*). Elle doit donc être interprétée avec prudence, car elle peut surestimer la performance sur les caractères musicaux réels.

### Conclusion

Sur les configurations testées, le LSTM avec gradient clipping constitue le meilleur modèle pour la prédiction du caractère suivant. Nous le retenons pour la génération de nouvelles partitions musicales dans l'exercice suivant.