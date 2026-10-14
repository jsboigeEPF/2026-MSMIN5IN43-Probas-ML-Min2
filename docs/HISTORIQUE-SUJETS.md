# Historique des sujets — volet IA probabiliste, théorie des jeux et machine learning

Inventaire des thèmes **déjà proposés ou déjà traités** dans les cours voisins de la promotion, pour éviter les doublons de verrou technique. Un sujet peut réapparaître sous une forme nouvelle à condition d'expliciter ce qui change (voir la règle de différenciation en fin de page). Cet historique ne contient aucun nom d'étudiant : les projets sont identifiés par leur thème et, quand c'est utile, par le nombre de groupes qui l'ont traité.

## Cours EPF 2025-2026 (MSMIN5IN43 — volet probabilités et apprentissage)

Thèmes **traités** (dossiers de groupe livrés dans le dépôt 2025) :

| Thème | Verrou technique traité en 2025 |
|---|---|
| TrueSkill / classement de joueurs | Inférence par graphes de facteurs, mise à jour de compétences en ligne |
| Media mix modeling (MMM) | Attribution publicitaire, saturation des dépenses, série temporelle |
| Prédictions sportives bayésiennes | Modèles de capacités joueur/équipe, calibration de prédictions |
| Poker de Kuhn (CFR) | Jeux à information incomplète, regret matching, équilibre approximé |
| Jeux évolutionnistes | Dynamique de réplicateurs, stabilité évolutionnaire |
| Détection de fake news (2 groupes) | Classifieurs texte, jeu de données public |
| Analyse de sentiment financier | NLP appliqué aux titres/actualités financières |
| Conformal prediction | Intervalles de confiance distribution-free |
| RL sur jeux simples (2 groupes : snake, pendule) | Q-learning / DQN de base |
| Génération de captchas | Génération d'images et adversarial de turing inversé |
| Chiffrement homomorphe | Calcul sur données chiffrées |
| Vision par ordinateur santé | Classification d'images médicales |

Ces verrous sont **fermés** : un projet 2026-2027 sur le même verrou est refusé ; un projet voisin doit citer sa différence (ex. « clustering bayésien non paramétrique » ≠ « sentiment de titres »).

## ECE 2026 — projet IA appliquée à la finance (livré)

| Thème | Nombre de groupes | Verrou technique |
|---|---:|---|
| Credit scoring et explicabilité | 3 | Classification déséquilibrée + SHAP sur données de crédit |
| Conformal prediction en finance | 3 | Intervalles conformes sur rendements/risque |
| Détection de fraude | 2 | Anomalie, classes déséquilibrées |
| RAG sur corpus financier | 2 | Retrieval-augmented generation |
| Portefeuille GNN | 1 | Graph neural networks pour la sélection d'actifs |
| Régimes de marché (Markov) | 1 | Modèles à états cachés / changement de régime |
| Pricing d'options | 1 | Méthodes numériques / ML pour la valorisation |
| Black-Litterman | 1 | Allocation combinant vues et équilibre de marché |
| Prévision de volatilité | 1 | Séries temporelles, GARCH/ML |
| PINN (physique-informé) | 1 | Réseaux guidés par des équations |
| Deep RL trading | 1 | Renforcement appliqué à l'exécution/le trading |
| Stratégie évolutionniste | 1 | Évolution / CMA-ES en finance |
| Données synthétiques par diffusion (tabulaires) | 1 | Génération de données tabulaires |
| ML causal en finance | 1 | CATE/ITE, causal forests, classement d'actifs par réactivité causale (la double différence n'y figurait qu'en perspective non implémentée) |

## EPITA 2025 — projet d'intelligence symbolique (verrou choix social)

| Thème | Verrou technique |
|---|---|
| Gouvernance multi-agents et vote | Méthodes de vote (Plurality, Borda, Condorcet, vote quadratique) et analyse de manipulabilité (vote stratégique, coalitions, corruption) |

## Cours MSMIS5IN11 2025 (majeure Ingénierie et santé — cours voisin, livré)

| Thème | Verrou technique |
|---|---|
| Réseau bayésien asthme | Graphe expert, inférence sur données cliniques |
| Cancer et explicabilité | XAI sur données cliniques |
| Simulation épidémique | Modèles compartimentaux, calibration |
| UNet pneumonie | Segmentation/classification d'images médicales |
| DCGAN médical | GAN pour images de synthèse |

Ces sujets relèvent du cours santé ; ils sont cités ici parce qu'un projet MSMIN5IN43 peut être tenté sur les mêmes verrous appliqués à un domaine non santé — la différenciation se joue alors sur le verrou, pas sur le domaine.

## Catalogue MSMIS5IN11 2026 (majeure santé — cours **concurrent**)

Le catalogue 2026-2027 de la majeure santé couvre déjà, entre autres : filtrage de Kalman, Dawid-Skene, change point, bandits, POMDP, valeur d'information, équité algorithmique, Stackelberg, persuasion, jeux à champ moyen, apprentissage fédéré, NSGA-II, calibration CMA-ES, effet causal d'un traitement sur données observationnelles (DoWhy, appariement, pondération, sensibilité). **Ces verrous sont réservés à l'autre cours** : ne pas les reprendre en 2026-2027 ici, même sur des données non santé.

## Règle de différenciation

Un sujet libre proche d'un thème ci-dessus n'est accepté que s'il **change le verrou technique**, pas seulement le nom ou le jeu de données :

- « Prédictions sportives bayésiennes » (2025) → fermé ; « modèle hiérarchique avec partial pooling évalué par prédiction croisée groupe » (A1) → verrou différent, accepté.
- « RL snake » (2025) → fermé ; « coopération émergente multi-agents avec mesures d'équité et dilemmes sociaux » (D3) → verrou différent, accepté.
- « Credit scoring + SHAP » (ECE 2026) → fermé ; le même pipeline sur un autre problème déséquilibré **avec matrice de coûts métier et seuil optimisé** (D1) → verrou différent, accepté, hors données de crédit.
- « Conformal prediction » (2025 et ECE 2026) → fermé dans toutes ses variantes simples.
- « Méthodes de vote à vainqueur unique et manipulabilité » (EPITA 2025) → fermé ; « élection d'un comité proportionnel, axiomes et calcul exact » (C2) → verrou différent, accepté.
- « ML causal en finance » (ECE 2026), « explications contrefactuelles non causales » (ECE 2026, IA explicable pour l'investissement, deux groupes) et « effet causal d'un traitement » (MSMIS5IN11 2026, réservé) → fermés ; « recours causal » (E1, mesure de la validité causale du recours) et « plans quasi expérimentaux » (E2) → verrous différents, acceptés.

En cas de doute, citez la ligne concernée de cet historique dans votre proposition et explicitez la différence : c'est le critère d'acceptation.
