from openai import OpenAI
from qdrant_client import QdrantClient, models
import os
import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import time
from functools import wraps
from typing import Callable, TypeVar, Any
import unicodedata

load_dotenv()

# Modification de la définition des types pour le décorateur
T = TypeVar('T')

def retry_with_backoff(max_attempts: int = 3, initial_delay: float = 1.0) -> Callable:
    """Décorateur qui réessaie une fonction avec un délai exponentiel.
    
    Args:
        max_attempts: Nombre maximum de tentatives
        initial_delay: Délai initial entre les tentatives en secondes
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            delay = initial_delay
            last_exception = None
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    last_exception = e
                    if attempt == max_attempts:
                        print(f"Échec après {max_attempts} tentatives. Dernière erreur: {str(e)}")
                        raise e
                    print(f"Tentative {attempt} échouée. Nouvelle tentative dans {delay} secondes...")
                    time.sleep(delay)
                    delay *= 2  # Délai exponentiel
            
            raise last_exception
        return wrapper
    return decorator

@retry_with_backoff(max_attempts=3, initial_delay=1.0)
def upsert_batch(client: QdrantClient, collection_name: str, points: list[models.PointStruct]) -> None:
    """Insère un lot de points dans Qdrant avec logique de réessai."""
    client.upsert(
        collection_name=collection_name,
        points=points
    )

# Configuration des clients
embedding_model = "text-embedding-3-small"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_API_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=300
)

# Configuration de la collection
qdrant_collection = "french"
qdrant_client.get_collection(collection_name=qdrant_collection)

# Configuration des chemins et compteurs
data_dir = os.path.join(os.path.dirname(__file__), "data")
vector_id_counter = 0


batch_points = []
document_count = 0

# Traitement des fichiers PDF
for file in os.listdir(f"{data_dir}"):
    if not file.lower().endswith('.pdf'):
        continue

    cleaned_file_name = unicodedata.normalize('NFKD', file).encode('ASCII', 'ignore').decode('ASCII')
    print(f'Traitement du fichier {cleaned_file_name}')

    # Lecture du PDF
    file_path = os.path.join(data_dir, file)
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        raise ValueError("Aucun texte trouvé dans le PDF")

    # Découpage du texte
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)

    # Création des embeddings
    for chunk in chunks:
        res = openai_client.embeddings.create(
            input=chunk,
            model=embedding_model
        )
        batch_points.append(models.PointStruct(
            id=vector_id_counter,
            vector=res.data[0].embedding,
            payload={
                "file": file,
                "text": chunk,
            }
        ))
        vector_id_counter += 1
        document_count += 1

    # Insertion par lots de 10 documents
    if document_count % 10 == 0:
        try:
            upsert_batch(qdrant_client, qdrant_collection, batch_points)
            print(f"Insertion de {len(batch_points)} vecteurs pour les 10 derniers documents.")
            batch_points = []
        except Exception as e:
            print(f"Échec de l'insertion du lot après toutes les tentatives : {str(e)}")
            continue

# Insertion des points restants
if batch_points:
    try:
        upsert_batch(qdrant_client, qdrant_collection, batch_points)
        print(f"Insertion de {len(batch_points)} vecteurs pour les documents restants dans le dossier {data_dir}.")
    except Exception as e:
        print(f"Échec de l'insertion du lot final après toutes les tentatives : {str(e)}")
