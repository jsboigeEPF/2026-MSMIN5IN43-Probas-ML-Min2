# Catalogue des sujets 2026-2027 — IA probabiliste, théorie des jeux et machine learning

Ce catalogue est propre à la promotion **MSMIN5IN43 2026-2027 (EPF, groupes Min1 et Min2)**. Il a été construit en trois temps : (1) compilation des sujets donnés aux promotions précédentes de ce cours et des cours voisins (EPF, ECE, EPITA) ; (2) retrait des sujets déjà traités par des groupes d'étudiants — voir [l'historique](HISTORIQUE-SUJETS.md) ; (3) complément par des sujets nouveaux. Aucun sujet du catalogue ne reprend le verrou technique d'un projet déjà rendu, et aucun ne figure au catalogue 2026-2027 du cours MSMIS5IN11 (majeure Ingénierie et santé), qui est un cours distinct.

Chaque fiche donne une difficulté indicative sur 5, des objectifs, les technologies clés et des liens **CoursIA** vers les notebooks du dépôt [jsboige/CoursIA](https://github.com/jsboige/CoursIA) qui servent de point de départ.

> **Exigence commune.** Un projet probabiliste ou ML se juge à ses **mesures** : baselines comparées, incertitudes quantifiées, expérience reproduite (graines fixées), et une analyse honnête des échecs. « Ça a l'air de marcher » n'est pas un résultat.
>
> **Données.** Jeux publics en accès ouvert ou synthétiques uniquement ; anonymisez ce qui doit l'être ; ne commitez pas les jeux téléchargés.

## Index

| Code | Sujet | Difficulté |
|---|---|:---:|
| **A** | **Inférence bayésienne et programmation probabiliste** | |
| A1 | Modèles hiérarchiques et partial pooling : quand les groupes partagent leurs connaissances | 3/5 |
| A2 | Segmenter sans connaître k : mélanges et clustering bayésien | 3/5 |
| A3 | Réseaux bayésiens : apprendre la structure, pas seulement les paramètres | 4/5 |
| A4 | Réseaux de neurones bayésiens : dire « je ne sais pas » | 4/5 |
| A5 | Processus gaussiens et optimisation bayésienne | 3/5 |
| **B** | **Décision, risque et actuariat** | |
| B1 | Assurance : fréquence-sévérité et crédibilité bayésienne | 3/5 |
| B2 | Utilité et décisions monétaires : primes de risque et paradoxes | 2/5 |
| **C** | **Théorie des jeux et mécanismes** | |
| C1 | Théorie de la négociation : marchandage de Nash et solutions axiomatiques | 3/5 |
| C2 | Élire un comité : règles proportionnelles, axiomes et calcul exact | 3/5 |
| C3 | Concevoir des enchères par apprentissage (mechanism design automatique) | 4/5 |
| C4 | Jeux sur graphes et contagion stratégique | 4/5 |
| **D** | **Machine learning rigoureux** | |
| D1 | Erreurs asymétriques : classes déséquilibrées, coûts et explication | 3/5 |
| D2 | Empoisonner un modèle : attaques et défenses par data poisoning | 4/5 |
| D3 | Apprentissage par renforcement multi-agents et coopération émergente | 4/5 |
| **E** | **Inférence causale** | |
| E1 | Recours causal : le changement recommandé inverse-t-il vraiment la décision ? | 4/5 |
| E2 | Identifier un effet sans randomisation : plans quasi expérimentaux | 4/5 |
| **F** | **Confidentialité et sécurité** | |
| F1 | Confidentialité différentielle : DP-SGD, budget et attaques d'inférence | 4/5 |
| F2 | Preuves zero-knowledge : prouver sans révéler | 4/5 |
| **G** | **Frontières de recherche** | |
| G1 | GFlowNets : générer proportionnellement à une récompense | 4/5 |
| G2 | Modèles de diffusion pour l'optimisation combinatoire | 4/5 |
| G3 | World models : apprendre un modèle de l'environnement | 4/5 |
| **H** | **Finance (un sujet, méthodologie exigeante)** | |
| H1 | Backtesting honnête : walk-forward, coûts et significativité | 3/5 |

---

## A — Inférence bayésienne et programmation probabiliste

### A1 — Modèles hiérarchiques et partial pooling : quand les groupes partagent leurs connaissances
**Difficulté : 3/5** · Mots-clés : `hiérarchique`, `partial pooling`, `PyMC`, `convergence`

Des données naturellement groupées (écoles, magasins, joueurs, régions) ne se résument ni par un modèle unique (on perd les différences) ni par un modèle par groupe (on sur-apprend). Le modèle hiérarchique partage l'information entre groupes. Le groupe de projet en construit un sur des données réelles et montre l'apport du pooling.
- **Objectifs** : jeu de données groupées public (résultats scolaires, e-commerce, sport, écologie) ; modèle complet (hyper-priorités, pentes variables) en PyMC ; comparaison systématique pooling total / partiel / aucun (prédiction Leave-One-Group-Out) ; diagnostics de convergence (R-hat, divergences) traités et documentés ; intervalle de crédibilité et intervalle fréquentiste comparés.
- **Technologies clés** : Python, PyMC, ArviZ.
- **CoursIA** : [Infer-12-Modeles-Hierarchiques](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-12-Modeles-Hierarchiques.ipynb) · [PyMC-12-Modeles-Hierarchiques](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-12-Modeles-Hierarchiques.ipynb) · [Infer-10-Model-Selection](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-10-Model-Selection.ipynb)
- **Références** : Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*, 2006 · McElreath, *Statistical Rethinking*, 2e éd. (bibliothèque du cours).

### A2 — Segmenter sans connaître k : mélanges et clustering bayésien
**Difficulté : 3/5** · Mots-clés : `mélanges`, `Dirichlet`, `non-paramétrique`, `EM vs MCMC`

Le k-means exige un k fixé et ignore l'incertitude. Les mélanges bayésiens estiment k (processus de Dirichlet), quantifient l'incertitude d'appartenance et gèrent les groupes vides. Le groupe compare ces régimes sur des données réelles.
- **Objectifs** : mise en œuvre EM (baseline) puis mélange gaussien bayésien et processus de Dirichlet (PyMC ou `sklearn` bayésien) ; critères de sélection (ELPD/WAIC) ; carte d'incertitude d'affectation ; étude de sensibilité aux priorités ; application thématique au choix (segments clients, styles d'écriture, genres musicaux).
- **Technologies clés** : Python, PyMC, scikit-learn, ArviZ.
- **CoursIA** : [Infer-2-Gaussian-Mixtures](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-2-Gaussian-Mixtures.ipynb) · [PyMC-02-Gaussian-Mixtures](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-02-Gaussian-Mixtures.ipynb) · [Infer-11-Topic-Models](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-11-Topic-Models.ipynb)
- **Références** : Neal, *Markov Chain Sampling Methods for Dirichlet Process Mixture Models*, JCGS, 2000 · Rasmussen, *The Infinite Gaussian Mixture Model*, NIPS, 1999.

### A3 — Réseaux bayésiens : apprendre la structure, pas seulement les paramètres
**Difficulté : 4/5** · Mots-clés : `structure learning`, `PC`, `hill climbing`, `inférence`

Un réseau bayésien spécifie un graphe causalement interprétable et des tables de probabilités conditionnelles. La plupart des usages fixent le graphe ; ici, le groupe **apprend la structure** à partir des données (contraintes, scores, approches hybrides) et en évalue la qualité.
- **Objectifs** : au moins deux algorithmes d'apprentissage de structure (PC ou contraintes, hill-climbing avec BIC/BDeu) sur des jeux publics ; inférence (élimination de variables, échantillonnage) sur le graphe appris ; évaluation : récupération des arêtes d'un graphe de référence (données synthétiques générées depuis un BN connu), stabilité par bootstrap ; comparaison avec une corrélation/lasso naïf ; discussion causalité vs corrélation (les arêtes apprises ne sont pas causales par défaut).
- **Technologies clés** : Python, `pgmpy` ou `bnlearn` (R via reticulate possible).
- **CoursIA** : [Infer-4-Bayesian-Networks](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-4-Bayesian-Networks.ipynb) · [PyMC-04-Bayesian-Networks](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-04-Bayesian-Networks.ipynb) · [Tweety-11-Causal-Python](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/SymbolicAI/Tweety/Tweety-11-Causal-Python.ipynb)
- **Références** : Koller & Friedman, *Probabilistic Graphical Models*, 2009 · Scutari, *Bayesian Networks with Examples in R*, 2021.

### A4 — Réseaux de neurones bayésiens : dire « je ne sais pas »
**Difficulté : 4/5** · Mots-clés : `incertitude épistémique`, `Variationnel`, `MC Dropout`, `OOD`

Un réseau classique donne la même confiance pour une donnée hors distribution que pour une donnée bien couverte. Les réseaux bayésiens (poids distribués, inférence variationnelle) séparent incertitude épistémique (données manquantes) et aléatoire (bruit). Le groupe implémente un petit BNN et mesure honnêtement ce qu'il apporte.
- **Objectifs** : BNN simple (1-2 couches) par inférence variationnelle (`Pyro` ou `TyXe`) et par MC Dropout (baseline) ; courbes précision vs confiance et diagrammes de fiabilité ; détection OOD mesurée (AUROC sur des jeux décalés) ; calibration (ECE) avant/après températures ; analyse critique : le gain vaut-il le coût de calcul multiplié ?
- **Technologies clés** : Python, PyTorch, Pyro ou TyXe ; `torchmetrics` pour l'étalonnage.
- **CoursIA** : [PyMC-09-Classification](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-09-Classification.ipynb) · [Infer-9-Classification](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-9-Classification.ipynb)
- **Références** : Blundell et al., *Weight Uncertainty in Neural Networks*, ICML, 2015 · Gal, *Uncertainty in Deep Learning*, thèse, 2016.

### A5 — Processus gaussiens et optimisation bayésienne
**Difficulté : 3/5** · Mots-clés : `GP`, `noyaux`, `acquisition`, `hyperparamètres`

Le processus gaussien régresse avec une incertitude analytique et un noyau qui encode la structure attendue ; l'optimisation bayésienne l'utilise pour optimiser une fonction coûteuse. Le groupe maîtrise les deux étages : régression GP, puis boucle d'acquisition sur une tâche réelle coûteuse.
- **Objectifs** : régression GP avec 3 noyaux comparés sur données bruitées (spatiales ou séries) ; effet des hyperparamètres du noyau (longueur d'échelle) visualisé ; optimisation bayésienne (EI, UCB) des hyperparamètres d'un modèle ML comparée à la recherche aléatoire et à la grille, budget d'évaluations fixé ; analyse du choix du noyau sur la convergence.
- **Technologies clés** : Python, scikit-learn (GP) ou GPyTorch, Optuna en comparaison.
- **CoursIA** : [Infer-16-Sparse-Gaussian-Process](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-16-Sparse-Gaussian-Process.ipynb) · [App-18-HyperparameterTuning](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Search/Applications/Hybrid/App-18-HyperparameterTuning.ipynb)
- **Références** : Rasmussen & Williams, *Gaussian Processes for Machine Learning*, 2006 · Snoek, Larochelle & Adams, *Practical Bayesian Optimization of ML Hyperparameters*, NeurIPS, 2012.

---

## B — Décision, risque et actuariat

### B1 — Assurance : fréquence-sévérité et crédibilité bayésienne
**Difficulté : 3/5** · Mots-clés : `fréquence/sévérité`, `crédibilité`, `bonus-malus`, `PyMC`

La prime d'assurance se décompose en fréquence de sinistres et coût moyen par sinistre — deux quantités que l'actuariat estime par des modèles bayésiens, avec un mécanisme de crédibilité qui mixe l'expérience individuelle et la moyenne du portefeuille. Le groupe construit ce pipeline sur des données ouvertes.
- **Objectifs** : jeu de données ouvert d'assurance (ex. « freMTPL2freq » ouvert via libraries R/OpenML, ou synthétique documenté) ; modèle fréquence (Poisson avec exposition) et sévérité (Gamma/lognormale) ; prime pure agrégée ; mécanisme de crédibilité type Bühlmann implémenté et comparé au posterior hiérarchique ; simulation de l'effet bonus-malus sur plusieurs années ; discussion de la charge de sécurité et de la réglementation.
- **Technologies clés** : Python, PyMC, pandas.
- **CoursIA** : [Actuariat-01-Prime-Pure-Chargement](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Actuariat/Actuariat-01-Prime-Pure-Chargement.ipynb) · [Actuariat-02-Freq-Sev-Hierarchique](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Actuariat/Actuariat-02-Freq-Sev-Hierarchique.ipynb) · [Actuariat-03-Actuarial-Credibility](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Actuariat/Actuariat-03-Actuarial-Credibility.ipynb)
- **Références** : Bühlmann & Gisler, *A Course in Credibility Theory*, 2005 · Frees, *Regression Modeling with Actuarial and Financial Applications*, 2009.

### B2 — Utilité et décisions monétaires : primes de risque et paradoxes
**Difficulté : 2/5** · Mots-clés : `utilité espérée`, `aversion au risque`, `assurance`, `paradoxe d'Allais`

Pourquoi assurer un téléphone mais pas sa maison contre une petite fuite ? La théorie de l'utilité espérée modélise ces choix. Le groupe construit un banc d'expérimentation : elicitation de fonctions d'utilité, calcul de primes de risque, et tests de paradoxes connus (Allais, réflexion) sur de vrais participants.
- **Objectifs** : elicitation de fonction d'utilité (méthode des équivalents certains) auprès de 15+ volontaires (consentement, anonymisation) ; vérification des axiomes et mise en évidence des violations ; calcul de prime d'équivalent certain et de prime de risque pour des loteries ; implémentation des paradoxes d'Allais et de la théorie des perspectives (Prospect Theory) en comparaison ; analyse des écarts entre participants.
- **Technologies clés** : Python, formulaires (Google Forms ou équivalent), statistiques.
- **CoursIA** : [DecInfer-01-Utility-Foundations](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/DecInfer/DecInfer-01-Utility-Foundations.ipynb) · [DecInfer-03-Utility-Money](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/DecInfer/DecInfer-03-Utility-Money.ipynb) · [DecInfer-04-Multi-Attribute](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/DecInfer/DecInfer-04-Multi-Attribute.ipynb)
- **Références** : von Neumann & Morgenstern, *Theory of Games and Economic Behavior*, 1944 · Kahneman & Tversky, *Prospect Theory*, Econometrica, 1979.

---

## C — Théorie des jeux et mécanismes

### C1 — Théorie de la négociation : marchandage de Nash et solutions axiomatiques
**Difficulté : 3/5** · Mots-clés : `marchandage`, `Nash`, `Kalai-Smorodinsky`, `expérimentation`

Deux agents doivent se partager un surplus : comment le répartir « équitablement » ? La théorie du marchandage (axiomatique et stratégique) propose plusieurs solutions. Le groupe les implémente, les compare en simulation, et les confronte à de la négociation réelle (humains ou agents LLM).
- **Objectifs** : ensemble de négociation et solutions de Nash, Kalai-Smorodinsky et utilitariste calculées sur des cas classiques (partage de gâteau, division d'un budget) ; dynamique stratégique (offres alternées de Rubinstein, équilibres sous-game perfect) ; expérience de négociation humaine par paires (protocole approuvé, anonymisé) ou agents LLM avec prompts de personnalité ; comparaison des accords observés aux solutions axiomatiques.
- **Technologies clés** : Python, `nashpy`, analyse de données expérimentales.
- **CoursIA** : [GameTheory-04d-Marchandage-Asymetrique](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-04d-Marchandage-Asymetrique-Python.ipynb) · [GameTheory-09-BackwardInduction](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-09-BackwardInduction-Python.ipynb)
- **Références** : Nash, *The Bargaining Problem*, Econometrica, 1950 · Rubinstein, *Perfect Equilibrium in a Bargaining Model*, Econometrica, 1982.

### C2 — Élire un comité : règles proportionnelles, axiomes et calcul exact
**Difficulté : 3/5** · Mots-clés : `multiwinner`, `Monroe`, `Chamberlin-Courant`, `représentation justifiée`, `ILP`

Élire k sièges parmi m candidats n'est pas élire un vainqueur k fois : les règles de comité (k-Borda, STV, Monroe, Chamberlin-Courant) répondent à des axiomes de proportionnalité différents — et les meilleures sont NP-difficiles à calculer exactement. Le groupe implémente, vérifie les axiomes et compare l'exact aux heuristiques.
- **Objectifs** : implémentation de k-Borda (baseline), STV, Monroe et Chamberlin-Courant, les deux derniers **exactement** par programme linéaire en nombres entiers (k et m modestes) ; axiomes de représentation (JR, EJR, PJR sur profils d'approbation) testés systématiquement par règle ; expériences : fréquence de violation des axiomes selon la distribution des préférences (impartiale, spatiale, polarisée) ; comparaison exact vs heuristique gloutonne en qualité et temps ; cas d'application : simulation d'assemblée ou recomptage d'une élection réelle publiée en open data si disponible.
- **Technologies clés** : Python, PuLP ou Z3 (optimisation), génération de profils.
- **CoursIA** : [SocialChoice/07-Committees-Core](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/SocialChoice/07-Committees-Core.ipynb) · [SocialChoice/09-Committees-STV-Monroe-ChamberlinCourant](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/SocialChoice/09-Committees-STV-Monroe-ChamberlinCourant.ipynb) · [SocialChoice/03-Voting-Methods](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/SocialChoice/03-Voting-Methods.ipynb) (rappel monocandidat)
- **Références** : Faliszewski et al., *Multiwinner Voting: A New Challenge for Social Choice Theory*, Trends Commun. Inf. Theory, 2017 · Aziz et al., *Justified Representation in Approval-Based Committee Voting*, Soc. Choice Welf., 2017 · Chamberlin & Courant, *Representative Deliberation and Representative Institutions*, APSR, 1983.

À distinguer du traitement 2025 (règles à vainqueur unique et analyse de manipulabilité) : ici le verrou est l'élection de **k sièges**, la proportionnalité et le calcul exact.

### C3 — Concevoir des enchères par apprentissage (mechanism design automatique)
**Difficulté : 4/5** · Mots-clés : `RegretNet`, `DSIC`, `différentiable`, `revenu`

Concevoir une règle d'enchère multi-objets qui maximise le revenu tout en dissuadant les menteurs est impossible analytiquement — mais le mécanisme peut être **appris** par descente de gradient, avec la contrainte d'incitation pénalisée dans la perte. Le groupe reproduit et critique cette approche.
- **Objectifs** : cadre théorique (enchère au second prix comme référence DSIC) ; implémentation de RegretNet (réseau allocation + paiements, pénalité de regret) sur des enchères à 2 objets et 3-5 acheteurs ; mesure du revenu et du regret empirique (stratégies optimales calculées par échantillonnage) ; comparaison aux enchères classiques (second prix indépendantes, VCG en petit) ; analyse : le mécanisme appris est-il vraiment DSIC ?
- **Technologies clés** : Python, PyTorch ; calcul modeste (CPU suffisant à petite échelle).
- **CoursIA** : [GameTheory-16-MechanismDesign](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-16-MechanismDesign-Python.ipynb) · [GameTheory-16b-Automated-Mechanism-Design](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-16b-Automated-Mechanism-Design-Python.ipynb) · [GameTheory-16c-Extraction-de-Revenu](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-16c-Extraction-de-Revenu-DSIC-IR-Python.ipynb)
- **Références** : Dütting et al., *Optimal Auctions through Deep Learning*, ICML, 2019 · Krishna, *Auction Theory*, 2009.

### C4 — Jeux sur graphes et contagion stratégique
**Difficulté : 4/5** · Mots-clés : `jeux de complémentarité`, `seuils`, `réseaux`, `dynamique de meilleure réponse`

Adopter une technologie, rejoindre une plateforme ou pratiquer une norme dépend de ce que font ses voisins : les choix sont **complémentaires** et se propagent sur un réseau. Le groupe étudie ces équilibres (seuils, cascades, blocage) par théorie et simulation.
- **Objectifs** : formalisation (jeux de complémentarité, équilibres de Nash sur graphe) ; algorithme de meilleure réponse itérée et détection de convergence/cycles ; calcul d'équilibres maximaux/minimaux (complémentarité → lattice) ; expériences sur graphes réels (open data de réseaux sociaux ou de collaboration) : effet de la structure (grappes, hubs) sur la cascade ; intervention : quels nœuds cibler pour déclencher/bloquer la contagion, comparé à des heuristiques de centralité.
- **Technologies clés** : Python, `networkx`, simulation à événements discrets.
- **CoursIA** : [GameTheory-02-NormalForm-Python](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-02-NormalForm-Python.ipynb) · [Dynamics-09-Biens-Publics-N-Personnes](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/Dynamics/Dynamics-09-Biens-Publics-N-Personnes-Python.ipynb) · [GameTheory-06-EvolutionTrust](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-06-EvolutionTrust-Python.ipynb)
- **Références** : Jackson, *Social and Economic Networks*, 2008 · Granovetter, *Threshold Models of Collective Behavior*, AJS, 1978.

---

## D — Machine learning rigoureux

### D1 — Erreurs asymétriques : classes déséquilibrées, coûts et explication
**Difficulté : 3/5** · Mots-clés : `déséquilibre`, `matrice de coûts`, `SHAP`, `seuil optimal`

Détecter 1 % de défaillances ou de fraudes avec une accuracy de 99 % est trivial et inutile : ce qui compte est le coût des faux négatifs contre les faux positifs. Le groupe construit une étude complète de bout en bout sur un problème réel de détection rare (maintenance, réseau, qualité), du choix de la métrique à l'explication. Le jeu de données du scoring crédit est exclu : il a servi trois fois en 2026 (voir [l'historique](HISTORIQUE-SUJETS.md)).
- **Objectifs** : pipeline complet sur un jeu déséquilibré public, hors scoring crédit (ex. détection de défauts, intrusion réseau, churn) ; baseline naïve majoritaire, puis modèles avec re-échantillonnage, pondération et seuillage ; matrice de coûts métier explicitée et seuil choisi par minimisation du coût (pas du F1 par défaut) ; explication SHAP/LIME des décisions et vérification de cohérence ; analyse d'erreurs chiffrée (quels sous-groupes, quels types).
- **Technologies clés** : Python, scikit-learn, XGBoost/LightGBM, `shap`.
- **CoursIA** : [2.12-Donnees-Desequilibrees](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours/2.12-Donnees-Desequilibrees.ipynb) · [2.14-Explicabilite-SHAP-LIME-Contrefactuels](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours/2.14-Explicabilite-SHAP-LIME-Contrefactuels.ipynb) · [2.13-Analyse-Erreurs](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours/2.13-Analyse-Erreurs.ipynb)
- **Références** : Elkan, *The Foundations of Cost-Sensitive Learning*, IJCAI, 2001 · Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS, 2017.

### D2 — Empoisonner un modèle : attaques et défenses par data poisoning
**Difficulté : 4/5** · Mots-clés : `data poisoning`, `backdoor`, `robustesse`, `défenses`

Un attaquant qui injecte ne serait-ce que 1 % de données d'entraînement corrompues peut installer une porte dérobée (backdoor) : le modèle est bon sur tout, sauf sur un déclencheur choisi. Le groupe met en œuvre ces attaques et les défenses connues, et mesure le trade-off.
- **Objectifs** : implémentation de 2-3 attaques (label flipping, backdoor BadNets avec déclencheur visuel, attaques proprement ciblées) sur un petit classifieur (MNIST/Fashion-MNIST ou texte) ; mesure de la réussite d'attaque et de la dégradation globale selon le taux d'empoisonnement ; défenses : filtrage par statistiques d'influence, apprentissage robuste (trimmed loss), détection de déclencheur par activation ; rapport honnête : quelles défenses tiennent, lesquelles s'effondrent.
- **Technologies clés** : Python, PyTorch, datasets standards ; éventuellement `detectron2`-style activation clustering simplifié.
- **CoursIA** : [4.6b-Detection-Adversarial-Camouflage](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/04-Vision/4.6b-Detection-Adversarial-Camouflage.ipynb) (robustesse adversariale en vision, contexte voisin) · [2.15-Donnee-Comme-Responsabilite](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours/2.15-Donnee-Comme-Responsabilite-Python.ipynb)
- **Références** : Gu, Dolan-Gavitt & Garg, *BadNets*, 2017 · Steinhardt, Koh & Liang, *Certified Defenses for Data Poisoning Attacks*, ICML, 2017.

### D3 — Apprentissage par renforcement multi-agents et coopération émergente
**Difficulté : 4/5** · Mots-clés : `MARL`, `IPPO`, `jeux coopératifs`, `dilemmes sociaux`

Plusieurs agents qui apprennent simultanément s'entraident ou se sabotent selon les incitations. Le groupe étudie la coopération émergente dans des dilemmes sociaux multi-agents (récolte partagée, nettoyage, switching de buts) avec des algorithmes d'apprentissage indépendant puis centralisé.
- **Objectifs** : environnement multi-agents (petit gridworld de dilemme social, maïs/harvest ou équivalent propre) ; IPPO/IA2C (indépendant) puis QMIX ou centralisé-critique ; mesures de coopération (rendement collectif, équité, exploitation des ressources) ; manipulation des incitations (récompense individuelle vs commune) et analyse des comportements ; étude de la non-stationnarité et de la catastrophe du commun dans les trajectoires d'apprentissage.
- **Technologies clés** : Python, PyTorch, RLlib ou implémentation propre à petite échelle.
- **CoursIA** : [GameTheory-17-MultiAgent-RL](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL-Python.ipynb) · [RL-07-Apprentissage-Multi-Agent](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/RL/RL-07-Apprentissage-Multi-Agent-Python.ipynb) · [RL-07b-Sur-Generalisation-Climbing-Game](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/RL/RL-07b-Sur-Generalisation-Climbing-Game-Python.ipynb)
- **Références** : Leibo, Zambaldi et al., *Multi-agent Reinforcement Learning in Sequential Social Dilemmas*, AAMAS, 2017 · Rashid et al., *QMIX*, ICML, 2018.

---

## E — Inférence causale

### E1 — Recours causal : le changement recommandé inverse-t-il vraiment la décision ?
**Difficulté : 4/5** · Mots-clés : `recours`, `contrefactuel individuel`, `abduction-action-prédiction`, `actionnabilité`

Un modèle refuse un prêt, une admission ou une subvention. Le recours algorithmique répond à « que puis-je changer pour que la décision s'inverse ? » — et cette réponse n'est causalement valable que si l'on sait raisonner en contrefactuels individuels (troisième échelon de l'échelle de Pearl). Le groupe construit un pipeline de recours complet et en mesure honnêtement la validité.
- **Objectifs** : classifieur entraîné sur un jeu public au choix (éducation, énergie, réadmissions, refus de prêt) ; génération de recours minimaux sous contraintes d'actionnabilité (certaines variables ne peuvent pas changer) ; deux régimes comparés — recours « boîte noire » (recherche locale autour de l'individu) et recours causaux (graphe causal explicite, procédure abduction-action-prédiction) ; **test de validité causale** : appliquer chaque recours recommandé (intervention réelle sur les variables du graphe) et mesurer la proportion de recours qui inversent réellement la décision contre celle des recours qui ne « marchent » que sur le classifieur ; sensibilité des recours aux hypothèses du graphe.
- **Technologies clés** : Python, DoWhy, EconML (ou CARLA), scikit-learn.
- **CoursIA** : [CausalBridges-01-Do-Calculus](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Causal-Bridges/CausalBridges-01-Do-Calculus.ipynb) · [CausalBridges-03-Dowhy-Contrefactuel-Individuel](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Causal-Bridges/CausalBridges-03-Dowhy-Contrefactuel-Individuel.ipynb) · [PyMC-05-Causal-Inference](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-05-Causal-Inference.ipynb)
- **Références** : Pearl, *Causality*, 2009 (ch. 7) · Wachter, Mittelstadt & Russell, *Counterfactual Explanations without Opening the Black Box*, Harvard JOLT, 2018 · Karimi et al., *Algorithmic Recourse: A Survey*, arXiv, 2020.

À distinguer de deux traitements ECE 2026 : les explications contrefactuelles **non causales** d'un projet d'IA explicable pour l'investissement (SHAP, LIME, changement minimal qui inverse la décision du classifieur), qui sont ici le **point de départ** et non le sujet, et le classement d'actifs par réactivité causale individuelle. À distinguer aussi du B1 du cours MSMIS5IN11 (effet causal **moyen** par ajustement sur facteurs de confusion) : ici le verrou est le recours actionnable et la **mesure de sa validité causale**.

### E2 — Identifier un effet sans randomisation : plans quasi expérimentaux
**Difficulté : 4/5** · Mots-clés : `variable instrumentale`, `discontinuité de régression`, `double différence`, `falsification`

Quand aucune expérience randomisée n'est possible, l'identification peut venir du **plan** : une variable instrumentale qui n'affecte l'issue qu'à travers le traitement, un seuil d'éligibilité (discontinuité de régression), ou une politique appliquée à certaines unités et pas à d'autres (double différence). Le groupe implémente au moins deux de ces plans sur des données publiques, avec leurs tests de falsification.
- **Objectifs** : variables instrumentales (2SLS, biais des instruments faibles, test de force instrumentale) ; discontinuité de régression (choix de bande passante, test de manipulation de la variable de forçage) ; double différence (hypothèse de tendance parallèle vérifiée, placebo sur période pré-traitement) ; au moins deux des trois plans sur des données publiques (open data éducation, emploi, environnement — ou le jeu de don d'organes fourni par la série Causal-Bridges) ; tests de falsification systématiques ; quand deux plans traitent la même question, comparaison des estimats et explication des écarts.
- **Technologies clés** : Python, `statsmodels`/`linearmodels`, DoWhy (modules instruments).
- **CoursIA** : [CausalBridges-06-Dowhy-Instrument-Faible](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Causal-Bridges/CausalBridges-06-Dowhy-Instrument-Faible.ipynb) · [CausalBridges-07-Quasi-Experimental](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Causal-Bridges/CausalBridges-07-Quasi-Experimental.ipynb)
- **Références** : Angrist & Pischke, *Mostly Harmless Econometrics*, 2009 · Imbens & Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*, 2015 · Card & Krueger, *Minimum Wages and Employment*, AER, 1994.

À distinguer du B1 du cours MSMIS5IN11 (ajustement sur facteurs de confusion) : ici l'identification vient du **plan** (instrument, seuil, double différence), pas de l'ajustement. La double différence n'apparaissait dans le projet ECE 2026 qu'en perspective non implémentée.

---

## F — Confidentialité et sécurité

### F1 — Confidentialité différentielle : DP-SGD, budget et attaques d'inférence
**Difficulté : 4/5** · Mots-clés : `différential privacy`, `DP-SGD`, `attaque d'appartenance`, `ε`

Un modèle peut mémoriser ses données d'entraînement : une attaque d'appartenance (membership inference) le démontre. La confidentialité différentielle borne ce risque — au prix de la précision. Le groupe met en scène l'attaque, la défense, et le compromis.
- **Objectifs** : attaque d'appartenance sur un modèle entraîné sans protection (mesure AUC d'attaque) ; implémentation de DP-SGD (Opacus) avec plusieurs budgets ε ; courbe utilité/privacy (accuracy vs force d'attaque) sur un jeu standard ; effet sur les sous-groupes rares ; discussion du choix d'un ε en pratique et des garanties formelles.
- **Technologies clés** : Python, PyTorch, Opacus ; `sklearn` pour l'attaquant.
- **CoursIA** : [2.15-Donnee-Comme-Responsabilite](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours/2.15-Donnee-Comme-Responsabilite-Python.ipynb) (responsabilité des données, cadre général)
- **Références** : Dwork & Roth, *The Algorithmic Foundations of Differential Privacy*, 2014 · Shokri et al., *Membership Inference Attacks Against ML Models*, IEEE S&P, 2017.

### F2 — Preuves zero-knowledge : prouver sans révéler
**Difficulté : 4/5** · Mots-clés : `ZK`, `Sigma protocoles`, `identité`, `vérification`

Prouver que l'on connaît un secret — un mot de passe, un solde suffisant, un âge supérieur à 18 — sans rien révéler de ce secret : les preuves zero-knowledge le permettent, avec des protocoles à base de théorie des groupes. Le groupe implémente des protocoles ZK classiques et une application complète.
- **Objectifs** : protocole de Fiat-Shamir (identité) et preuve de connaissance d'un isomorphisme de graphe, implémentés de zéro ; démonstrateur interactif (prover/verifier) avec taux d'erreur sonore mesuré ; application au choix : vérification d'âge, vote à bulletin vérifiable simplifié, ou loterie équitable prouvée ; transformation non-interactive (Fiat-Shamir heuristique) ; analyse des hypothèses cryptographiques.
- **Technologies clés** : Python, `sympy`/arithmétique modulaire, éventuellement `py_ecc`.
- **CoursIA** : [SC-15-Zero-Knowledge-Proofs](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/SymbolicAI/SmartContracts/04-Privacy-Cryptography/SC-15-Zero-Knowledge-Proofs-Python.ipynb) · [SC-17-E2E-Verifiable-Voting](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/SymbolicAI/SmartContracts/04-Privacy-Cryptography/SC-17-E2E-Verifiable-Voting-Python.ipynb)
- **Références** : Goldwasser, Micali & Rackoff, *The Knowledge Complexity of Interactive Proof Systems*, STOC, 1985 · Fiat & Shamir, *How to Prove Yourself*, CRYPTO, 1986.

---

## G — Frontières de recherche

### G1 — GFlowNets : générer proportionnellement à une récompense
**Difficulté : 4/5** · Mots-clés : `GFlowNet`, `échantillonnage`, `objets composés`, `récompense`

Un GFlowNet apprend une politique qui **construit** des objets composés (séquences, graphes, molécules simplifiées) avec une probabilité proportionnelle à leur récompense — échantillonner la diversité, pas seulement l'optimum. Le groupe implémente un GFlowNet simple et le compare aux méthodes usuelles.
- **Objectifs** : cadre théorique (trajectoires, loss de trajectoire-balance) ; implémentation sur un environnement de grille ou de génération de petites chaînes/set (bibliothèque `torchgfn`) ; comparaison avec échantillonnage par rejet, MCMC et politique RL (PPO) sur la même récompense : diversité des modes trouvés, correspondance à la distribution cible (KL) ; analyse de l'effet de la température ; extension : génération de graphes simples (motifs).
- **Technologies clés** : Python, PyTorch, `torchgfn`.
- **CoursIA** : [3.6-Modeles-Generatifs](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/03-DeepLearning/3.6-Modeles-Generatifs.ipynb) (modèles génératifs, socle)
- **Références** : Bengio et al., *Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation*, NeurIPS, 2021 · documentation [torchgfn](https://github.com/GFNOrg/torchgfn).

### G2 — Modèles de diffusion pour l'optimisation combinatoire
**Difficulté : 4/5** · Mots-clés : `diffusion`, `TSP`, `solutions discrètes`, `guidage`

Les modèles de diffusion apprennent à reconstruire des données à partir de bruit — y compris des **solutions** de problèmes combinatoires. Le groupe étudie cette approche (type DIFUSCO) sur un problème bien maîtrisé (TSP ou coloration de petites instances), entraîne un petit modèle et le compare aux heuristiques. À distinguer de la synthèse de données tabulaires par diffusion, déjà traitée en 2026.
- **Objectifs** : compréhension du cadre (diffusion discrète sur matrices d'adjacence) ; entraînement d'un petit modèle (UNet léger ou GNN) sur des solutions optimales d'instances TSP générées (solveur exact ou LK pour les instances d'entraînement) ; échantillonnage guidé et arrêt avant le bruit complet ; comparaison aux heuristiques (plus proche voisin, 2-opt) : qualité, temps, robustesse à la taille ; discussion : généralisation à des tailles non vues.
- **Technologies clés** : Python, PyTorch ; génération d'instances ; GPU utile mais pas indispensable à petite échelle.
- **CoursIA** : [3.10-Modeles-Generatifs-Diffusion-SOTA](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/03-DeepLearning/3.10-Modeles-Generatifs-Diffusion-SOTA.ipynb) · [Frontieres-10-NeuralDiving-Coloration](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Search/Part5-Frontieres/Frontieres-10-NeuralDiving-Coloration-Python.ipynb)
- **Références** : Sun et Yang, *DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization*, NeurIPS, 2023 · Applegate et al., TSPLIB / solveur Concorde (génération d'optimaux).

### G3 — World models : apprendre un modèle de l'environnement
**Difficulté : 4/5** · Mots-clés : `model-based RL`, `Dreamer`, `planification dans le rêve`

Au lieu d'apprendre une politique directement, un world model apprend la **dynamique** de l'environnement et planifie « dans son imagination ». Le groupe implémente une version simplifiée (type Dreamer ou Dyna) et quantifie le gain d'échantillons par rapport au RL sans modèle.
- **Objectifs** : environnement visuel simple (cart-pole pixel ou mini-grid) ; implémentation d'un Dyna (modèle appris de transitions + planification) puis d'un world model latent simplifié (encodeur + dynamique + décodeur) ; comparaison sample-efficiency contre DQN/PPO sans modèle (mêmes budgets d'interactions réelles) ; étude des « hallucinations » du modèle (où le monde appris se trompe) et de leur effet sur la politique ; visualisation de l'espace latent.
- **Technologies clés** : Python, PyTorch, `gymnasium`.
- **CoursIA** : [RL-16-Dream-RSI](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/RL/RL-16-Dream-RSI-Python.ipynb) · [RL-08-Dyna-Q-Planification](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/RL/RL-08-Dyna-Q-Planification-Python.ipynb)
- **Références** : Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer), ICLR, 2020 · Ha & Schmidhuber, *World Models*, NeurIPS, 2018.

---

## H — Finance (un sujet, méthodologie exigeante)

### H1 — Backtesting honnête : walk-forward, coûts et significativité
**Difficulté : 3/5** · Mots-clés : `walk-forward`, `coûts de transaction`, `significativité`, `quantconnect`

Ce sujet ne promet pas de battre le marché : il apprend à **ne pas se mentir** en backtest. Fuite de données, survie des meilleures stratégies, coûts ignorés, un seul jeu de test : le groupe construit un protocole d'évaluation rigoureux pour quelques stratégies simples et montre comment chaque piège gonfle artificiellement la performance.
- **Objectifs** : 3-4 stratégies simples (momentum, moyenne mobile, portefeuille parcimonieux) sur un même univers d'actifs ; protocole walk-forward strict (fenêtres glissantes, hyperparamètres choisis uniquement sur le passé) ; intégration de coûts réalistes (spread, commission, impact) ; tests de significativité (t-stats avec correction des tests multiples, p-values boostrappées) ; rapport final : performance dégradée par chaque garde-fou, et verdict honnête par stratégie.
- **Technologies clés** : Python, pandas, `vectorbt` ou backtrader ; éventuellement [QuantConnect](https://www.quantconnect.com/) LEAN pour validation finale.
- **CoursIA** : [App-27-Sparse-Index-Tracking-Walk-Forward](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Search/Applications/Hybrid/App-27-Sparse-Index-Tracking-Walk-Forward.ipynb)
- **Références** : López de Prado, *Advances in Financial Machine Learning*, 2018 (ch. backtesting) · Bailey et al., *The Probability of Backtest Overfitting*, J. Comput. Finance, 2017.
