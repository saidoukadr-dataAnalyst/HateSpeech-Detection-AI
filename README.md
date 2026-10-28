# Détection de discours haineux avec validation humaine

Projet académique de Deep Learning (S8, UIR) réalisé par Mahamat Saleh Touma, Aymane Aitouhammou, Said Oukadr et Achraf Houcine Akkouch, encadré par Hakim Hafidi.

Le prototype associe une classification binaire PyTorch, une vérification lexicale et un agent CrewAI chargé de synthétiser les résultats. Un opérateur peut approuver ou rejeter le rapport ; sa décision est enregistrée dans un journal JSON.

## Fonctionnement du code actuel

1. `data_cleaner.py` nettoie les textes des fichiers Parquet (colonnes `text` et `HS`).
2. `train_model.py` construit une représentation TF-IDF de 5 000 caractéristiques au maximum et entraîne un MLP : entrée → 256 → 128 → 2, ReLU et dropout 0,5. L'entraînement utilise Adam, 10 époques et des lots de 64.
3. `main.py` expose deux outils : inférence PyTorch et recherche de mots dans un lexique anglais. Un seul agent hybride CrewAI utilise ces outils avec le modèle Groq configuré `llama-3.3-70b-versatile`.
4. La CLI demande une validation humaine. `app.py` expose aussi les routes Flask `/`, `/analyze` et `/validate`.

## État et limites

- Les poids et le vectoriseur sont présents localement. Aucun score n'a été reproduit lors de cet examen ; le script d'entraînement calcule une accuracy et une matrice de confusion.
- Le template `templates/index.html` est absent : la page d'accueil Flask ne peut pas fonctionner en l'état.
- Le nettoyage utilisé à l'inférence diffère de celui de l'entraînement.
- Le lexique et les stop words sont anglais ; la qualité multilingue n'est pas établie.
- Le rapport final décrit une architecture multi-agent plus large que l'implémentation actuelle.
- Les versions des dépendances ne sont pas verrouillées et le correctif interne CrewAI doit être vérifié avec la version installée.
- Les fichiers de données nécessitent une provenance et des conditions de redistribution documentées avant publication.

## Exécution locale

Dépendances utilisées : Python, torch, scikit-learn, pandas, un moteur Parquet comme pyarrow, crewai, Flask et requests (diagnostic Groq). L'environnement n'a pas été validé de bout en bout.

Depuis ce dossier, définir une nouvelle clé dans la session PowerShell :

```powershell
$env:GROQ_API_KEY = "votre-cle"
python main.py
```

`python train_model.py` réentraîne le modèle et remplace les poids et le vectoriseur locaux. Les textes soumis à l'agent sont transmis au service Groq et peuvent être enregistrés dans le journal local.

## Préparation de la publication

Les clés intégrées au code ont été remplacées par la variable d'environnement `GROQ_API_KEY`. Révoquer l'ancienne clé chez Groq. Ne pas publier l'archive ZIP d'origine, qui contient une ancienne copie du projet. Le `.gitignore` exclut l'environnement, les journaux, données et artefacts locaux ; aucun dépôt distant n'est créé par cette préparation.

Les sources se trouvent dans le dossier `DL_PROJECT/`. Exécuter les commandes Python depuis ce sous-dossier.
