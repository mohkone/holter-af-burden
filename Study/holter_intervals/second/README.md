# Second détecteur, ArNet2

Le second détecteur est le réseau neuronal ArNet2 de Biton et coll., appliqué avec les poids distribués par les auteurs. Son entrée est une série d’intervalles entre battements automatiques. La prédiction par fenêtre est la classe binaire que renvoie la signature publique `predict_fixed`. Le modèle a été développé dans la cohorte UVAF de l’Université de Virginie. La cohorte japonaise SHDB faisait partie des évaluations externes des auteurs. Ceux-ci ont étudié un résultat incluant fibrillation et flutter, alors que la cible de notre étude est le temps annoté en fibrillation seule. Des populations ayant beaucoup de flutter peuvent donc présenter des erreurs systématiques. [Publication](https://doi.org/10.1038/s41746-023-00791-1) · [Dépôt officiel](https://github.com/aim-lab/AFtoolkit/tree/aa48a24c34095c8749e977ec5ad8b1442717c81b/detection)

## Calcul

1. Les temps des battements sont convertis en intervalles RR en secondes à la fréquence de chaque enregistrement. Les doublons conservent un intervalle nul. La source des instants automatiques est documentée dans chaque fiche de cohorte. Pour LTA, seuls les événements automatiques N du fichier QRS sont des battements. Les marques T manuelles sont ignorées. Chaque patient et chaque enregistrement constituent une séquence réseau distincte.
2. Les RR successifs sont divisés en fenêtres de 60 intervalles sans chevauchement, depuis le premier battement valide. Les intervalles d’une fenêtre complète sont transmis tels quels au modèle. Les métadonnées du modèle sont l’index de fenêtre partant de zéro, un identifiant constant dans l’enregistrement et la constante 1 attendue par le lanceur des auteurs. Aucune annotation de rythme, de diagnostic ou de résultat individuel n’intervient.
3. Pour obtenir un score à la minute, on interroge la classe de la fenêtre contenant les instants 5, 15, 25, 35, 45 et 55 secondes. Un instant hors de toute fenêtre complète, ou situé dans un intervalle RR de moins de 0,25 s ou plus de 2,5 s, contribue pour 0,5. Chaque minute entière reçoit la moyenne de ces six votes, éventuellement sur treize niveaux espacés de 1/12. Cette projection temporelle et cette règle de qualité ont été fixées pour le pilote. Les durées mesurées restent sujettes aux erreurs des détecteurs automatiques de battements et à l’irrégularité non liée à la fibrillation. Les masques d’éligibilité des minutes de référence s’appliquent ensuite dans l’analyse du sondage.
4. Les tableaux `derived/COHORTE/IDENTIFIANT.npz` contiennent l’index de minute à partir de zéro, le score et la proportion de votes indéterminés. Chaque manifeste donne les empreintes SHA-256, le nombre de minutes et le nombre de battements pour les enregistrements traités.

## Reproduction du réseau

La reproduction statistique du rapport utilise les scores de minutes fournis. Pour regénérer ces scores sur CPU, un environnement distinct peut installer `tensorflow-cpu==2.18.1` avec Python 3.12. L’inférence a été effectuée sous Linux x86-64. Les poids du modèle de l’AIM Lab sont publiés sous CC BY-NC 4.0 ; le fichier de licence original est `ARNET2_LICENSE.md`. Le module de téléchargement contrôle la taille, l’empreinte SHA-256 et l’identité Git de chaque fichier. [Version figée](https://github.com/aim-lab/AFtoolkit/tree/aa48a24c34095c8749e977ec5ad8b1442717c81b/detection/exported_model)

Depuis ce répertoire :

```bash
python fetch_arnet.py --output exported_model
python arnet.py --pilot /chemin/vers/holter_pilot --cohort LTA
python arnet.py --pilot /chemin/vers/holter_pilot --cohort SHDB
python predict_new.py --source ../cohorts/derived --cohort IRIDIA
```

La variable d’environnement `ARNET2_MODEL_DIR` peut désigner un autre dossier de poids vérifiés. Pour LTA et SHDB, le pilote initial doit contenir les sources automatiques QRS et les en-têtes originaux. Pour les nouvelles cohortes, seuls les trois champs `peaks`, `fs` et `n_samples` sont utilisés. Les poids TensorFlow ne sont pas nécessaires pour consulter les résultats déposés.

## Vérifications

Le test `test_arnet.py` compare la projection par minute à un calcul scalaire sur des fréquences et rythmes synthétiques, avec des doublons, des lacunes et des fenêtres incomplètes. Le test indépendant `verify_upstream.py` appelle le lanceur original sur deux séquences synthétiques à 128 et 200 Hz. Les 160 classes de fenêtres et les 110 scores de minute sont identiques à notre implémentation. Les données de vérité de ces cohortes n’ont pas servi à ces vérifications. La commande du contrôle original requiert pandas et le lanceur officiel de la révision indiquée. Son identifiant Git est enregistré dans `upstream_runner.json`.
