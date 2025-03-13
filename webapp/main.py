from flask import Flask, render_template, request, jsonify
from qdrant_client import QdrantClient
import os
from openai import OpenAI
import dotenv
import logging
from waitress import serve

dotenv.load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration des clients
embedding_model = "text-embedding-3-small"
llm_model = "gpt-4o-mini" # Utilisation d'un modèle plus puissant
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_API_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=300
)

# Configuration de la collection
qdrant_collection = "french"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    # Récupérer la requête de l'utilisateur
    user_query = request.json.get('query', '')
    
    if not user_query:
        return jsonify({"error": "Requête vide"}), 400
    
    # Convertir la requête en vecteur en utilisant OpenAI
    response = openai_client.embeddings.create(
        input=user_query,
        model=embedding_model
    )
    query_vector = response.data[0].embedding
    
    # Rechercher dans Qdrant avec un nombre plus élevé de résultats
    search_result = qdrant_client.search(
        collection_name=qdrant_collection,
        query_vector=query_vector,
        limit=8  # Augmentation du nombre de documents récupérés
    )
    
    # Extraire les résultats et préparer le contexte
    results = []
    context = ""
    
    # Logging des documents récupérés
    logger.info("=== DOCUMENTS RÉCUPÉRÉS DE QDRANT ===")
    
    for i, res in enumerate(search_result):
        # Filtrer les résultats avec un score trop bas
        if res.score < 0.3:  # Seuil de pertinence
            continue
            
        result_data = {
            "score": res.score,
            "payload": res.payload
        }
        results.append(result_data)
        
        # Log du document
        logger.info(f"Document {i+1} - Score: {res.score}")
        if "text" in res.payload:
            doc_content = res.payload["text"]
            logger.info(f"Contenu: {doc_content[:200]}... (tronqué)")
            # Ajouter le contenu au contexte avec des métadonnées
            context += f"--- DOCUMENT {i+1} (PERTINENCE: {res.score:.2f}) ---\n{doc_content}\n\n"
        else:
            logger.info(f"Payload sans champ 'text': {res.payload}")
    
    logger.info("===============================")
    
    # Vérifier si le contexte est vide
    if not context.strip():
        logger.warning("ATTENTION: Le contexte est vide! Aucun document avec champ 'text' trouvé.")
    
    # Générer une réponse avec OpenAI en utilisant le contexte - Prompt amélioré pour une meilleure présentation
    prompt = f"""
Tu es un assistant spécialisé dans l'univers de Warhammer 40,000 et particulièrement Kill Team.
Ta mission est de fournir des informations précises et détaillées sur les règles, les factions, les stratégies et le lore.

Utilise UNIQUEMENT les informations contenues dans les documents fournis ci-dessous pour répondre à la question.
Si les documents ne contiennent pas suffisamment d'informations pour répondre complètement, indique clairement les limites de ta réponse.
N'invente JAMAIS d'informations qui ne sont pas présentes dans les documents.

DOCUMENTS DE RÉFÉRENCE:
{context}

QUESTION: {user_query}

INSTRUCTIONS DE FORMATAGE:
1. Commence par une réponse concise (1-2 phrases) qui résume l'essentiel
2. Structure ta réponse avec des titres en gras (utilise ** pour le gras)
3. Utilise des listes à puces pour les énumérations (points clés, étapes, options)
4. Limite chaque paragraphe à 2-3 phrases maximum
5. Utilise des sauts de ligne fréquents pour aérer le texte
6. Si tu mentionnes des valeurs numériques importantes (points de vie, coûts, etc.), mets-les en gras
7. Pour les règles complexes, utilise ce format: "Règle: [explication simple]"
8. Termine par une conclusion pratique ou un conseil tactique en une phrase

EXEMPLE DE STRUCTURE:
**Réponse rapide**: [1-2 phrases résumant l'essentiel]

**Détails principaux**:
• Point clé 1
• Point clé 2

**Informations complémentaires**:
Explication courte et claire.

**Conclusion**: Conseil tactique ou résumé pratique.
"""
    
    # Logging des informations envoyées à OpenAI
    logger.info("=== INFORMATIONS ENVOYÉES À OPENAI ===")
    logger.info(f"Question utilisateur: {user_query}")
    logger.info(f"Nombre de documents récupérés: {len(results)}")
    logger.info(f"Taille du contexte: {len(context)} caractères")
    logger.info("===============================")
    
    messages = [
        {"role": "system", "content": "Tu es un expert de Kill Team. Tu fournis des réponses précises, détaillées et dans le style de l'univers du 41ème millénaire. Pour l'Empereur!"},
        {"role": "user", "content": prompt}
    ]
    
    # Utilisation d'un modèle plus puissant avec des paramètres optimisés
    chat_completion = openai_client.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=0.3,  # Température plus basse pour des réponses plus précises
        max_tokens=300,  # Limite de tokens plus élevée pour des réponses détaillées
        top_p=0.95,
        presence_penalty=0.1,
        frequency_penalty=0.1
    )
    
    ai_response = chat_completion.choices[0].message.content
    
    # Logging de la réponse
    logger.info("=== RÉPONSE D'OPENAI ===")
    logger.info(ai_response)
    logger.info("===============================")
    
    # Retourner à la fois les résultats de recherche et la réponse générée
    return jsonify({
        "results": results,
        "ai_response": ai_response,
        "debug_info": {
            "context_length": len(context),
            "num_documents": len(results),
            "has_content": bool(context.strip()),
            "model_used": llm_model
        }
    })

# Point d'entrée pour l'exécution avec Waitress
def create_app():
    return app

if __name__ == '__main__':
    # Configuration pour le développement
    if os.getenv("FLASK_ENV") == "development":
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Configuration pour la production avec Waitress
        port = int(os.getenv("PORT", 5000))
        logger.info(f"Démarrage du serveur Waitress sur le port {port}")
        serve(app, host='0.0.0.0', port=port, threads=4)
