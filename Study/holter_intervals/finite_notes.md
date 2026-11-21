# Intervalles pour la moyenne finie d’un Holter

Les minutes éligibles, leurs fractions de fibrillation et les scores du détecteur sont considérés comme fixes. Dans chacune des strates disjointes les minutes sont tirées au hasard simple sans remise, indépendamment des autres strates, selon un effectif fixé avant la lecture. Les résultats ci-dessous portent sur les nouveaux tirages de vingt minutes dans ce même Holter. Les deux constructions acceptent toute fraction de temps comprise entre zéro et un, y compris les minutes de transition.

Une strate entièrement relue apporte exactement sa somme divisée par le nombre total de minutes. On la retire de l’inférence, puis on ajoute sa contribution aux deux bornes finales. Dans les expressions ci-dessous, les poids W et les moyennes sont renormalisés sur la masse non recensée. Une fois les bornes calculées, on les multiplie par cette masse avant d’ajouter la contribution exacte. Une strate non vide dotée de zéro lecture n’est pas admise par le plan. Les deux fonctions renvoient immédiatement le résultat exact si le Holter est entièrement relu.

## Borne directe de Chernoff

Écrire n_h pour les effectifs échantillonnés, p_h pour les moyennes de population, x_h pour les moyennes échantillonnées et W_h pour les poids démographiques, de somme un. Posons x = somme W_h x_h, p = somme W_h p_h, w_h = W_h/n_h et b = max w_h. Pour tout lambda réel, le logarithme de la fonction génératrice de x vérifie

    log E exp(lambda*x) ≤ (1/b) log(1 − p + p exp(lambda*b)).

La comparaison convexe de Hoeffding entre les tirages sans et avec remise borne d’abord la fonction génératrice de chaque strate par un produit de n_h facteurs. L’inégalité de corde de l’exponentielle sur [0,1] majore chacun par 1−p_h+p_h exp(lambda*w_h). La convexité en w du logarithme de ce facteur et sa valeur nulle en w=0 permettent de le majorer, sur l’échelle logarithmique, par w_h/b fois sa valeur en b. Les strates indépendantes s’additionnent sur cette échelle. La concavité du logarithme en p_h achève la majoration.

Pour chaque sens de déviation, la borne de Chernoff est exp[−D(x || p)/b], où D est la divergence de Bernoulli. L’intervalle observé inverse D(x || p) ≤ b log(2/alpha). Il est généralement conservateur et n’utilise pas les prédictions dans son calcul après la définition du plan. Référence de la comparaison sans remise : [Hoeffding (1963), théorème 4](https://doi.org/10.1080/01621459.1963.10500830).

## Intervalle projeté utilisant les prédictions

Fixer avant le tirage q_h, la moyenne des prédictions calibrées de toutes les minutes de la strate, ramenée dans [0,01 ; 0,99]. Pour une valeur hypothétique p_h et une minute de charge y, comparer les deux facteurs

    G(y) = (q_h/p_h)^y ((1−q_h)/(1−p_h))^(1−y)
    A(y) = y q_h/p_h + (1−y)(1−q_h)/(1−p_h).

L’inégalité des moyennes arithmétique et géométrique donne G(y) ≤ A(y). Lorsque p_h est la vraie moyenne, la moyenne finie de A(y) vaut un. Le produit de n_h valeurs A(y) tirées sans remise a une espérance au plus égale à un, par l’inégalité de Maclaurin sur la moyenne symétrique élémentaire normalisée. En multipliant les produits des strates indépendantes, l’espérance du produit global des G reste au plus égale à un. Aux limites p_h=0 ou 1, le vrai contenu d’une strate est déterministe et la même conclusion s’obtient par continuité sur son support. L’emploi de q_h appris sur les étiquettes effectivement relues dans ce même tirage nécessiterait un autre argument.

L’inégalité de Markov donne une région simultanée de niveau au moins 1−alpha sur le vecteur des moyennes. La région, exprimée à partir des seules moyennes des échantillons, est

    somme n_h D(x_h || p_h) ≤ log(1/alpha) + somme n_h D(x_h || q_h).

Les deux extrêmes de somme W_h p_h sur cette région constituent l’intervalle pour la charge de l’enregistrement. La région contient les moyennes échantillonnées. Elle reste valide pour des q_h imprécis. Une mauvaise concordance de q_h avec la population peut augmenter sensiblement sa largeur. Le niveau de la région est conjoint, avec un facteur 1/alpha unique pour les deux extrêmes.

Pour calculer le maximum, le multiplicateur t ≥ 0 permet de résoudre séparément chaque variable. Pour z=t W_h/n_h, la racine pertinente de z p_h²+(1−z)p_h−x_h=0 appartient à [0,1]. Lorsque x_h=0 elle vaut max(0,1−1/z) pour z>0. Une recherche scalaire trouve la valeur de t qui sature la contrainte de divergence. L’algorithme prend pour la borne haute le côté extérieur de l’encadrement numérique. Le minimum se déduit par complémentarité en remplaçant chaque x_h par 1−x_h.

L’inversion d’un produit de tests de moyennes locales et sa projection sur une moyenne stratifiée ont déjà été développées. Il s’agit ici d’une spécialisation à date fixe avec prédictions auxiliaires gelées. Pour les constructions par paris et inférence stratifiée, voir [Waudby-Smith et Ramdas (2024)](https://doi.org/10.1093/jrsssb/qkad009) et [Spertus, Sridhar et Stark, Sequential stratified inference for the mean](https://arxiv.org/abs/2409.06680).

## Vérification

Le script test_finite.py vérifie les cas limites analytiques, l’inversion scalaire indépendante, 50 extrêmes comparés à un optimiseur générique et les recensements exacts. Un contrôle de couverture sur 7 200 échantillons exhaustifs dans des populations à fractions intermédiaires vérifie aussi le premier moment du produit de paris. Un calcul hypergéométrique sur 5 280 couples population–intervalle à deux strates, avec plusieurs prédictions gelées, donne une couverture minimale de 94,75 % au niveau nominal de 90 %. Ces contrôles testent l’implémentation. La garantie générale provient des inégalités précédentes. Résultats dans results/finite_tests.txt.
