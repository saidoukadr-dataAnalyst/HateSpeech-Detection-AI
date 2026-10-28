Rapport de Projet :
Détection de Discours Haineux avec Deep Learning et IA Agentique



Cours: Deep Learning (S8 UIR)

Sujet : Construction d'un pipeline complet d'analyse sémantique, de l'apprentissage profond à la supervision par agents IA.






Réalisation de :
                     Mahamat Saleh Touma 
                     Aymane Aitouhammou
                     Said Oukadr
                    Achraf Houcine Akkouch


Encadré par Mr : Hakim Hafidi













1. Introduction et Contexte

La prolifération d'Internet et l'essor des réseaux sociaux ont drastiquement modifié la manière dont les individus interagissent. Cependant, cette liberté d'expression s'est accompagnée d'une montée alarmante des discours haineux (hate speech) et de la toxicité en ligne. La modération manuelle de ces contenus est aujourd'hui impossible en raison du volume astronomique de données générées quotidiennement.

C'est dans ce contexte que s'inscrit ce projet de Deep Learning. L'objectif principal est de concevoir un système automatisé hybride capable non seulement de détecter les propos haineux avec précision, mais également de fournir un raisonnement explicable et d'impliquer une supervision humaine. 

Notre solution repose sur une architecture moderne en trois temps :
1. **Un réseau de neurones profond (PyTorch) :** Spécialement entraîné pour classer les textes sur des bases statistiques.
2. **Une intelligence artificielle agentique (CrewAI) :** Agissant comme un orchestrateur cognitif qui combine les prédictions du réseau de neurones avec des analyses linguistiques déterministes.
3. **Une interface Human-In-The-Loop (HITL) :** Développée avec Flask, permettant à un opérateur humain de valider ou rejeter les prédictions, garantissant ainsi l'éthique et le contrôle du système.

Ce rapport détaille les choix architecturaux, le traitement des données, la modélisation mathématique et l'intégration logicielle de l'ensemble du pipeline.

---


































2. Nettoyage et Préparation des Données

La qualité des données est le pilier central de tout projet d'apprentissage automatique. Pour ce projet, nous avons utilisé un dataset contenant des milliers de textes classifiés, incluant des tweets et des commentaires potentiellement offensants. 

### 2.1. Le Pipeline de Nettoyage
Le texte brut extrait d'Internet est souvent très bruité. Il contient des liens hypertexte, des mentions d'utilisateurs, de la ponctuation excessive et des caractères spéciaux qui n'apportent aucune valeur sémantique pour la classification des discours haineux.

Nous avons développé un module de nettoyage (`data_cleaner.py`) basé sur des expressions régulières (Regex) avec Pandas :

```python
import pandas as pd
import re

def clean_tweet_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # URLs
    text = re.sub(r'\@\w+', '', text) # Mentions
    text = re.sub(r'\#', '', text) # Hashtags
    text = re.sub(r'[^\w\s]', '', text) # Ponctuation
    return text.lower().strip()
```
Ces étapes permettent de réduire l'espace dimensionnel du vocabulaire et d'éviter à notre modèle de sur-apprendre (overfitting) sur des éléments non pertinents comme la typographie.

### 2.2. Vectorisation TF-IDF
Les réseaux de neurones ne comprennent pas le texte sous forme de chaînes de caractères ; ils exigent des tenseurs mathématiques. Nous avons opté pour la méthode **TF-IDF (Term Frequency - Inverse Document Frequency)**.

Contrairement au simple encodage *Bag of Words*, le TF-IDF pénalise les mots qui apparaissent trop fréquemment dans l'ensemble du corpus (comme "the", "and") et donne plus de poids aux mots rares et porteurs de sens (comme "hate", "scum"). Le vectoriseur, limité à un maximum de caractéristiques, transforme ainsi chaque texte en un vecteur creux (sparse vector). Ce vectoriseur est ensuite sauvegardé via `pickle` pour garantir que les textes soumis en production seront encodés avec le même vocabulaire que lors de l'entraînement.

---















3. Architecture et Entraînement du Modèle PyTorch

Le cœur statistique de notre application est un réseau de neurones artificiels (ANN) construit avec le framework PyTorch. 

### 3.1. Choix de l'Architecture
Nous avons implémenté un **Perceptron Multicouche (MLP)**. Bien que les réseaux récurrents (RNN) ou les Transformers soient populaires en NLP, un MLP robuste combiné à un bon encodage TF-IDF s'avère extrêmement rapide et performant pour la classification binaire.

Notre classe `HateSpeechMLP` hérite de `nn.Module` et se compose de :
*   **Couche d'entrée :** Correspondant à la taille du vocabulaire TF-IDF.
*   **Couches cachées :** Deux couches denses (Linear) de 256 et 128 neurones.
*   **Activation ReLU :** Pour introduire la non-linéarité essentielle à l'apprentissage complexe.
*   **Dropout (0.5) :** Une technique de régularisation forte où 50% des neurones sont désactivés aléatoirement pendant l'entraînement. Cela force le réseau à répartir l'apprentissage et prévient le surentraînement.
*   **Couche de sortie :** 2 neurones (classification binaire : Haineux / Non-Haineux).

```python
class HateSpeechMLP(nn.Module):
    def __init__(self, input_dim):
        super(HateSpeechMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.out = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.out(x)
```

### 3.2. Optimisation et Entraînement
Nous avons utilisé la fonction de perte `CrossEntropyLoss` (idéale pour la classification multiclasses/binaire) combinée à l'optimiseur **Adam**, qui adapte dynamiquement le taux d'apprentissage.
Durant l'entraînement, les données sont injectées par `DataLoader` sous forme de tenseurs. À chaque époque, la passe avant (forward pass) calcule les prédictions, la fonction de coût évalue l'erreur, et la passe arrière (backpropagation) met à jour les poids.
À l'issue de l'entraînement, le modèle compilé est sauvegardé dans le fichier `hate_speech_model.pth`.

---











4. Le Système Multi-Agent avec CrewAI

Un réseau de neurones produit une probabilité mathématique (ex: 95% de chance que le texte soit toxique). Cependant, dans le cadre d'un système de modération professionnel, une probabilité brute n'est pas suffisante. Il est nécessaire de pouvoir **expliquer** la décision.

C'est ici qu'intervient **CrewAI**, un framework permettant de créer des agents autonomes basés sur de grands modèles de langage (LLMs). Nous utilisons l'API Groq (via le modèle Llama) pour animer un "cerveau" analytique.

### 4.1. L'Approche par Outils (Tooling)
Nous avons encapsulé nos algorithmes dans des classes héritant de `BaseTool` de CrewAI :
1.  **PyTorchInferenceTool :** Charge le vecteur, charge le modèle `.pth`, passe le texte de l'utilisateur dans le modèle, et renvoie la probabilité (le résultat du Deep Learning).
2.  **ContextCheckerTool :** Effectue une passe linguistique déterministe (par N-Grams) pour vérifier la présence factuelle de mots proscrits dans un lexique spécifique.

### 4.2. L'Agent Hybride
Initialement, nous avions conçu un système à trois agents (un expert DL, un linguiste, un orchestrateur). Toutefois, les interactions inter-agents consommaient trop de tokens (jetons), entraînant des erreurs de quota "Rate Limit" (12 000 TPM sur Groq).
Pour contourner cette limitation tout en conservant l'intelligence du système, nous avons implémenté le pattern architectural de **l'Agent Hybride** :

```python
hybrid_analyzer_agent = Agent(
    role='Super Analyste (Lexique & Deep Learning)',
    goal="Analyser le texte à l'aide de ContextCheckerTool ET PyTorchInferenceTool, puis donner un court verdict final.",
    backstory="Expert hybride. Tu utilises tes outils pour extraire les faits, puis tu rédiges une conclusion en 2 lignes.",
    tools=[pytorch_inference_tool, context_checker_tool],
    llm=llm
)
```
Cet agent agit comme un modérateur virtuel. Il reçoit un texte, utilise l'outil linguistique pour repérer les insultes, utilise l'outil PyTorch pour obtenir la probabilité statistique, puis croise ces informations pour rédiger un rapport intelligible ("Le texte est haineux avec 99% de confiance, notamment à cause du mot X repéré").

---
















 5. Interface Utilisateur et "Human-In-The-Loop" (HITL)

L'IA n'est pas infaillible (risques de faux positifs ou de biais). L'architecture **Human-In-The-Loop** remet l'humain au centre de la décision critique.

### 5.1. Serveur Web (Flask)
Nous avons déployé un serveur Flask (`app.py`) agissant comme backend. Il expose des routes `/analyze` (qui appelle CrewAI) et `/validate` (qui enregistre la décision finale). Ce micro-framework assure une communication fluide via requêtes AJAX/JSON avec le navigateur de l'utilisateur.

### 5.2. UI Design et Expérience Utilisateur
L'interface (frontend) a été conçue sans utiliser de frameworks lourds, en s'appuyant sur du Vanilla CSS avec une esthétique moderne **Glassmorphism** (fonds semi-transparents, effets de flou "backdrop-filter", palettes sombres et accents néon violet). 
L'utilisateur entre un texte, visualise l'état de réflexion de l'agent en direct (spinner), puis lit le rapport final synthétique.

### 5.3. Audit et Logs
Une fois le rapport affiché, l'opérateur humain clique sur "Approuver" ou "Rejeter". Cette action déclenche l'écriture dans un fichier `hate_speech_logs.json`.
Ces logs sont inestimables. Ils ne servent pas qu'à la traçabilité ; ils constituent une base de données qualifiée (données + prédiction + correction humaine) qui pourra être injectée à l'avenir dans le réseau de neurones PyTorch pour déclencher un **réentraînement continu (Active Learning)**.

---
































6. Conclusion et Perspectives

Ce projet démontre qu'il est possible de dépasser les modèles de machine learning isolés pour créer un véritable produit logiciel intelligent et éthique. 

**Réussites du projet :**
*   Entraînement réussi d'un modèle PyTorch personnalisé sur un jeu de données textuelles complexes.
*   Pionnier dans l'utilisation de CrewAI pour orchestrer l'inférence d'un réseau de neurones local, apportant de l'explicabilité aux "boîtes noires" du Deep Learning.
*   Implémentation réussie de la boucle HITL et contournement pragmatique des limitations d'API par refactorisation agentique.

**Perspectives (Futurs travaux) :**
1.  **Amélioration du Modèle :** Remplacer le MLP par un modèle Transformer (comme BERT ou RoBERTa) fine-tuné, qui capte le contexte de la phrase entière au lieu des fréquences de mots simples.
2.  **Base de données :** Migrer le système de log JSON vers une base de données relationnelle robuste (PostgreSQL) couplée à un pipeline MLOps automatisé pour relancer l'entraînement du réseau la nuit sur les logs corrigés par les humains.
3.  **Déploiement Cloud :** Conteneuriser l'application sous Docker pour un déploiement scalable sur des services comme AWS ou Google Cloud.

