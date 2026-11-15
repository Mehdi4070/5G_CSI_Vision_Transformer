# Script oral à quatre intervenants — 5G CSI Vision Transformer

Durée visée : **13 à 15 minutes**, hors questions.

## Répartition proposée

| Intervenant | Diapositives | Partie | Durée indicative |
|---|---:|---|---:|
| **Mehdi** | 1 à 4 | Introduction, système et principe du CSI | environ 3 min 10 |
| **Personne 2** | 5 à 7 | Scénario, prétraitement et architectures | environ 3 min 35 |
| **Personne 3** | 8 à 11 | Protocole, résultats, multipath et bruit | environ 4 min 10 |
| **Personne 4** | 12 à 15 | Paramètres, discussion et conclusion | environ 3 min 15 |

Remplacer « Personne 1 », etc. par vos prénoms. Chaque intervenant doit
connaître la conclusion générale et les chiffres principaux, même lorsqu'ils
ne figurent pas dans sa propre partie.

Le texte ci-dessous est volontairement plus complet que ce qu'il faut
mémoriser mot à mot. Il vaut mieux retenir l'idée de chaque paragraphe et parler
naturellement.

# Intervenant 1 — Introduction et principe

## Diapositive 1 — Titre — 20 secondes

> Bonjour. Nous allons vous présenter notre projet intitulé « Localisation des
> équipements utilisateurs basée sur le CSI dans les réseaux 5G ». L'objectif
> est d'utiliser l'information d'état du canal comme une empreinte radio, puis
> d'apprendre la relation entre cette empreinte et la position d'un utilisateur
> avec un Vision Transformer.

Transition :

> Je commence par préciser le problème et les objectifs expérimentaux.

## Diapositive 2 — Problématique et objectifs — 55 secondes

> Pour chaque utilisateur, nous disposons d'une matrice CSI complexe organisée
> selon les antennes de la station de base et les sous-porteuses OFDM. Nous
> cherchons à prédire directement les coordonnées continues x et y. Il s'agit
> donc d'un problème de régression, pas de classification.
>
> Le CSI est intéressant parce qu'il dépend fortement de la position : les
> distances, les angles, les réflexions et les retards changent lorsque
> l'utilisateur se déplace. Nos objectifs sont de générer les données avec
> DeepMIMO, de construire un prétraitement sans fuite, de comparer CNN, ViT et
> modèle hybride, puis d'étudier le multipath, le bruit et les paramètres du
> système.

Transition :

> Avant de détailler le CSI, voici la chaîne complète que nous avons mise en
> œuvre.

## Diapositive 3 — Étapes principales — 55 secondes

> Notre système comporte sept étapes. Nous générons d'abord les canaux avec
> DeepMIMO, puis nous validons les dimensions, les positions et les valeurs.
> Le CSI complexe est ensuite converti en tenseur réel. Nous réalisons un
> découpage spatial en zones d'entraînement, de validation et de test, puis
> nous ajustons la normalisation uniquement sur le train.
>
> Les mêmes données servent à entraîner le CNN, le ViT et l'architecture
> hybride. Enfin, nous évaluons les modèles en mètres et réalisons les études
> complémentaires sur le multipath, le bruit, les antennes et les
> sous-porteuses. Cette séparation des étapes assure la reproductibilité et
> évite les fuites d'information.

Transition :

> Pour comprendre pourquoi cette chaîne peut estimer une position, regardons
> ce que contient exactement une CSI.

## Diapositive 4 — Contenu du CSI — 60 secondes

> La CSI d'un utilisateur est ici une matrice complexe 64 par 64. Chaque ligne
> correspond à une antenne et chaque colonne à une sous-porteuse. Un coefficient
> possède une amplitude et une phase.
>
> Les 64 antennes apportent de la diversité spatiale et contiennent notamment
> de l'information sur les angles d'arrivée. Les 64 sous-porteuses apportent de
> la diversité fréquentielle et permettent d'observer les effets des différents
> retards. L'amplitude décrit principalement l'atténuation, tandis que la phase
> est liée à la distance et aux trajets de propagation.
>
> La structure complète peut donc être considérée comme une empreinte radio de
> la position, beaucoup plus riche qu'une simple mesure RSSI.

Transition :

> Nous savons maintenant pourquoi le CSI peut servir d'empreinte de position.
> Je laisse la parole à **[prénom de la personne 2]**, qui va présenter le
> scénario, le prétraitement et les modèles étudiés.

# Intervenant 2 — Données, prétraitement et modèles

## Diapositive 5 — Scénario et dataset DeepMIMO — 70 secondes

> Merci. Le dataset est produit avec DeepMIMO à partir du scénario O1_28. O1 désigne
> la géométrie d'un environnement extérieur urbain avec des rues et des
> bâtiments, tandis que 28 indique la fréquence porteuse de 28 GHz, dans la
> bande FR2. Nous utilisons le jeu d'émetteurs 4 et la grille de récepteurs 0.
> Dans cette grille, nous sélectionnons une zone d'environ 20 mètres par 20
> mètres contenant 10 000 utilisateurs, avec une position tous les 20
> centimètres et une hauteur de 2 mètres.
>
> La station utilise un réseau 8 par 8, donc 64 antennes espacées d'une
> demi-longueur d'onde. Le système possède 64 sous-porteuses sur 50 MHz et
> retient jusqu'à 10 trajets. Après génération, nous avons vérifié la forme, les
> positions, les valeurs non finies, les CSI nuls et les doublons. Les 10 000
> exemples sont valides.
>
> La station paraît éloignée parce qu'elle est installée hors du rectangle des
> utilisateurs. Cette situation est normale : la figure montre seulement la
> petite région étudiée et non un rayon de couverture. Le ray tracing calcule
> les trajets entre la station et chaque utilisateur de cette région.

Transition :

> Le CSI brut étant complexe, une transformation est nécessaire avant
> l'apprentissage.

## Diapositive 6 — Prétraitement et split — 70 secondes

> Nous convertissons chaque CSI en trois canaux réels : amplitude logarithmique
> en dB, cosinus de la phase et sinus de la phase. Le couple cosinus-sinus évite
> la discontinuité entre moins pi et plus pi.
>
> Chaque canal est standardisé avec une moyenne et un écart-type calculés
> uniquement sur les utilisateurs d'entraînement. Les coordonnées sont aussi
> standardisées. Utiliser les statistiques du test serait une fuite
> d'information.
>
> Le découpage est spatial et non aléatoire. Nous utilisons 6 800 utilisateurs
> pour l'entraînement, 1 400 pour la validation, 1 400 pour le test et 400 comme
> zones de garde. Un split aléatoire placerait des positions séparées de 20 cm
> entre train et test et produirait une performance trop optimiste.

Transition :

> À partir de ces tenseurs, nous comparons trois familles de modèles.

## Diapositive 7 — Architectures — 75 secondes

> Le premier modèle est un CNN compact de 86 290 paramètres. Il applique quatre
> blocs convolutionnels et apprend principalement des motifs locaux.
>
> Le modèle principal est un Vision Transformer. La matrice 64 par 64 est
> découpée en patches 8 par 8, ce qui produit 64 tokens. Nous ajoutons un token
> de régression, puis quatre blocs Transformer avec quatre têtes d'attention et
> une dimension d'embedding de 96. Les positions des antennes et des
> sous-porteuses utilisent deux embeddings distincts, car ces deux axes ont une
> signification physique différente.
>
> Enfin, le modèle hybride commence par un stem CNN puis applique l'attention
> sur les caractéristiques réduites. Il cherche à combiner extraction locale et
> dépendances globales.

Transition :

> Ces trois architectures sont maintenant prêtes à être entraînées et
> comparées. Je passe la parole à **[prénom de la personne 3]**, qui va
> présenter le protocole et les résultats expérimentaux.

# Intervenant 3 — Protocole et résultats expérimentaux

## Diapositive 8 — Protocole — 45 secondes

> Merci. Les modèles prédisent des coordonnées normalisées et sont entraînés avec une
> MSE. Le ViT utilise AdamW avec un learning rate initial de 3 fois 10 puissance
> moins 4, un scheduler et un early stopping. La graine est fixée à 42 et le
> meilleur ViT est obtenu à l'époque 26.
>
> Pour l'évaluation, les coordonnées sont reconverties en mètres. Nous calculons
> l'erreur euclidienne, sa moyenne, sa médiane, la RMSE, P90, P95 et la CDF. Tous
> les modèles utilisent les mêmes 1 400 utilisateurs de test.

Transition :

> Voici maintenant la comparaison principale.

## Diapositive 9 — Résultats principaux — 75 secondes

> La baseline naïve, qui prédit toujours la position moyenne du train, obtient
> plus de 10 mètres d'erreur. Le CNN descend à 1,607 mètre. Le Vision
> Transformer obtient la meilleure moyenne, 0,970 mètre, ainsi qu'une médiane de
> 0,883 mètre et un P90 de 1,794 mètre. Il réduit donc l'erreur moyenne du CNN de
> 39,6 pour cent.
>
> La CDF confirme que son avantage ne vient pas uniquement de quelques exemples :
> sa courbe est globalement la plus à gauche. L'hybride obtient 1,818 mètre. Son
> downsampling convolutionnel peut avoir supprimé des détails fins avant
> l'attention. Cela ne signifie pas que tous les hybrides sont mauvais, mais que
> cette configuration précise n'est pas optimale.

Transition :

> Après cette comparaison, nous avons étudié l'effet des paramètres de
> propagation.

## Diapositive 10 — Multipath — 65 secondes

> Nous avons généré quatre variantes retenant au maximum 1, 3, 5 ou 10 trajets,
> avec les mêmes utilisateurs, le même split, le même ViT et la même graine.
>
> Un trajet donne la meilleure erreur, 0,715 mètre. Trois trajets donnent 0,749,
> cinq trajets 1,612 et dix trajets 0,970. La variation n'est donc pas monotone.
> Un canal simple est plus facile à inverser, mais davantage de trajets peuvent
> aussi fournir des informations supplémentaires.
>
> Comme nous n'avons utilisé qu'une graine, la conclusion prudente est que la
> richesse du multipath modifie fortement la difficulté. Nous ne devons pas
> affirmer que moins de trajets est toujours meilleur.

Transition :

> Nous avons ensuite perturbé le CSI pour mesurer la robustesse des modèles.

## Diapositive 11 — Bruit — 65 secondes

> Un bruit blanc gaussien complexe est ajouté directement au CSI, avant le
> prétraitement. Les modèles sont gelés et ont été entraînés uniquement sur des
> données propres. Chaque SNR est évalué avec trois réalisations indépendantes.
>
> À 20 dB, le ViT reste pratiquement au niveau propre, avec 0,997 mètre. À 10 dB,
> il passe à 1,358 mètre. Même à 0 dB, lorsque bruit et signal ont la même
> puissance, il reste le meilleur avec 3,767 mètres, contre 9,341 pour le CNN.
>
> L'agrégation globale peut aider à exploiter les parties moins perturbées de la
> matrice. Cela reste une interprétation plausible et non une preuve causale.

Transition :

> Le ViT conserve donc son avantage lorsque le CSI est bruité. Pour terminer
> l'analyse, je laisse la parole à **[prénom de la personne 4]**, qui va
> présenter l'influence des dimensions du système, les limites et la
> conclusion.

# Intervenant 4 — Analyse finale et conclusion

## Diapositive 12 — Antennes et sous-porteuses — 75 secondes

> Merci. Pour les antennes, nous comparons des sous-réseaux centrés 4 par 4, 4 par 8 et
> 8 par 8, en conservant 64 sous-porteuses. L'erreur passe de 2,305 mètres avec
> 16 antennes à 1,306 avec 32, puis 0,970 avec 64.
>
> Pour l'étude fréquentielle, nous conservons les 64 antennes et sélectionnons
> 16, 32 ou 64 sous-porteuses uniformément sur la même bande de 50 MHz. Les
> erreurs sont respectivement 1,512, 1,401 et 0,970 mètre.
>
> Les deux dimensions sont utiles. Mais pour le même nombre de coefficients,
> garder les antennes et réduire les fréquences est moins pénalisant. Dans ce
> scénario, la diversité spatiale semble donc particulièrement importante.

Transition :

> Il reste important de préciser ce que ces résultats permettent réellement de
> conclure.

## Diapositive 13 — Discussion et limites — 65 secondes

> Le premier enseignement méthodologique est l'importance du split spatial. Le
> second est que le ViT exploite efficacement la structure globale du CSI et
> conserve son avantage sous bruit. Enfin, antennes et sous-porteuses apportent
> des informations complémentaires.
>
> Cependant, l'étude utilise un seul scénario simulé, une seule station, une
> hauteur utilisateur fixe et une seule graine d'entraînement. Le bruit AWGN ne
> représente pas tous les défauts matériels, et nous n'avons pas encore testé
> des CSI réelles.
>
> La formulation correcte est donc : le ViT est le meilleur modèle sur notre
> protocole contrôlé. Nous ne prétendons pas qu'il est universellement supérieur
> dans tous les environnements.

Transition :

> Je termine avec le bilan et les perspectives.

## Diapositive 14 — Conclusion — 55 secondes

> Nous avons construit une chaîne complète allant du ray tracing à la position
> 2D, avec 10 000 utilisateurs et un découpage spatial strict. Le ViT atteint
> une erreur moyenne de 0,970 mètre et une médiane de 0,883 mètre. Il surpasse le
> CNN, l'hybride et la baseline naïve, et il est également le plus robuste au
> bruit.
>
> Les prochaines étapes seraient de répéter les entraînements avec plusieurs
> graines, d'ajouter du bruit pendant l'entraînement, de tester la
> généralisation entre scénarios et surtout d'utiliser des mesures CSI réelles
> et des utilisateurs mobiles.
>
> Le message principal est que le CSI peut être exploité comme un véritable
> capteur de position intégré au réseau 5G.

Transition :

> Merci pour votre attention. Nous sommes prêts à répondre à vos questions.

## Diapositive 15 — Questions

Les quatre intervenants regardent le jury et attendent la question. La personne
la plus à l'aise sur le sujet concerné répond en premier, puis un collègue peut
compléter sans répéter la même réponse. Garder à
portée la banque de questions/réponses et mémoriser au minimum les chiffres
suivants : **10 000 utilisateurs, 6 800/1 400/1 400, CSI 64×64, ViT 0,970 m,
CNN 1,607 m, médiane ViT 0,883 m, P90 1,794 m**.

## Conseils de présentation

- Ne pas lire les tableaux ligne par ligne : annoncer uniquement la tendance et
  le meilleur résultat.
- Toujours préciser « dans notre scénario » pour les conclusions sur le
  multipath et les antennes.
- Ne pas dire que 0,970 m signifie que tous les utilisateurs sont localisés à
  moins d'un mètre ; le P90 est 1,794 m.
- Si une réponse n'a pas été vérifiée expérimentalement, dire clairement que
  c'est une hypothèse ou une perspective.
- Pendant les questions, commencer par la réponse directe, puis donner
  l'explication.
