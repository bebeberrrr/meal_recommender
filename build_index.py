import pandas as pd
import os
import shutil
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import sys

# --- SETUP LOGGING ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# --- 1. CONFIGURE FILE PATHS AND MODELS ---
INPUT_FILE = "cleaned_recipes2_processed.csv"
INDEX_DIR = "./recipes_index"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- 2. DELETE OLD INDEX ---
if os.path.exists(INDEX_DIR):
    print(f"Deleting old index at {INDEX_DIR}...")
    shutil.rmtree(INDEX_DIR)
    print("Old index deleted.")

try:
    # --- 3. CONFIGURE LLAMAINDEX SETTINGS ---
    print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    # This sets the embedding model for all LlamaIndex operations
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = None
    print("Embedding model configured.")

    # --- 4. LOAD DATA AND CREATE 'DOCUMENTS' ---
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Fill any NaNs in text fields to prevent errors
    df['name'] = df['name'].fillna('')
    df['cuisinetype'] = df['cuisinetype'].fillna('')
    df['ingredients_text'] = df['ingredients_text'].fillna('')
    df['combined_text'] = df['combined_text'].fillna('')

    print("Converting DataFrame rows to LlamaIndex Documents...")
    documents = []

    for _, row in df.iterrows():
        doc_text = row['combined_text']

        metadata = {
            "name": row['name'],
            "ingredients": row['ingredients_text'],
            "cuisinetype": row['cuisinetype'],
            "calories": float(row.get('calories', 0.0)),
            "fatcontent": float(row.get('fatcontent', 0.0)),
            "saturatedfatcontent": float(row.get('saturatedfatcontent', 0.0)),
            "cholesterolcontent": float(row.get('cholesterolcontent', 0.0)),
            "sodiumcontent": float(row.get('sodiumcontent', 0.0)),
            "carbohydratecontent": float(row.get('carbohydratecontent', 0.0)),
            "fibercontent": float(row.get('fibercontent', 0.0)),
            "sugarcontent": float(row.get('sugarcontent', 0.0)),
            "proteincontent": float(row.get('proteincontent', 0.0))
        }

        doc = Document(text=doc_text, metadata=metadata)
        documents.append(doc)

    print(f"Successfully created {len(documents)} Documents.")

    # --- 5. BUILD AND SAVE THE INDEX ---
    print("Building new index from documents...")
    print("This may take a few minutes...")

    # This step creates the vectors and builds the index
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )

    # Save the index to disk
    print(f"Saving index to {INDEX_DIR}...")
    index.storage_context.persist(persist_dir=INDEX_DIR)

    print("\n--- Indexing Complete ---")
    print(f"New index is built and saved in '{INDEX_DIR}'.")
    print("You can now start api_app.py.")

except FileNotFoundError:
    print(f"❌ ERROR: File not found at '{INPUT_FILE}'.")
    print("Please make sure you have run preprocess_data.py first.")
except Exception as e:
    print(f"An error occurred: {e}")