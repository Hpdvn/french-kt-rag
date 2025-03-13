# Kill Team RAG - Système d'Intelligence Tactique

![Kill Team Logo](https://d1w82usnq70pt2.cloudfront.net/wp-content/uploads/2021/08/killteam-logo.png)

## À propos

Kill Team RAG est un système d'intelligence artificielle spécialisé dans l'univers de Warhammer 40,000 Kill Team. Cette application utilise la technologie RAG (Retrieval-Augmented Generation) pour fournir des réponses précises et contextuelles aux questions sur les règles, les factions, les stratégies et le lore de Kill Team.

## Fonctionnalités

- **Recherche sémantique** : Trouve les informations pertinentes grâce aux embeddings vectoriels
- **Réponses intelligentes** : Génère des réponses précises basées sur les sources officielles
- **Interface thématique** : Design inspiré de l'univers de Warhammer 40,000
- **Support Markdown** : Mise en forme riche des réponses pour une meilleure lisibilité
- **Optimisé pour mobile** : Expérience utilisateur fluide sur tous les appareils

## Technologies utilisées

- **Backend** : Python, Flask
- **Base de données vectorielle** : Qdrant
- **IA générative** : OpenAI GPT
- **Frontend** : HTML5, CSS3, JavaScript
- **Embeddings** : OpenAI text-embedding-3-small

## Architecture

L'application utilise une architecture RAG (Retrieval-Augmented Generation) :

1. Les questions de l'utilisateur sont converties en vecteurs d'embedding
2. Ces vecteurs sont utilisés pour rechercher les documents pertinents dans Qdrant
3. Les documents récupérés servent de contexte pour générer une réponse précise via OpenAI
4. La réponse est formatée en Markdown et présentée à l'utilisateur

## Installation

### Prérequis

- Python 3.9+
- Compte OpenAI avec clé API
- Instance Qdrant (cloud ou locale)

### Configuration

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-username/killteam-rag.git
   cd killteam-rag
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Créez un fichier `.env` à la racine du projet :
   ```
   OPENAI_API_KEY=votre_clé_api_openai
   QDRANT_API_URL=votre_url_qdrant
   QDRANT_API_KEY=votre_clé_api_qdrant
   ```

### Lancement

```bash
python webapp/main.py
```

L'application sera accessible à l'adresse `http://localhost:5000`.

## Utilisation

1. Entrez votre question sur Kill Team dans le champ de recherche
2. Cliquez sur "INTERROGER" ou appuyez sur Entrée
3. Consultez la réponse générée et les sources utilisées
4. Utilisez le bouton "RESET" pour effacer les résultats et poser une nouvelle question

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Remerciements

- Games Workshop pour l'univers de Warhammer 40,000 et Kill Team
- OpenAI pour leur API de génération de texte
- Qdrant pour leur base de données vectorielle

---
