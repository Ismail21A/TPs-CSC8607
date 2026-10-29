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

