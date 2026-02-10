MYRAGE: A Retrieval-Augmented Generation Framework for Food Recommendation
Project Overview
The MYRAGE framework is a Retrieval-Augmented Generation (RAG) system designed to optimize meal planning and recipe selection.
By leveraging a high-performance Large Language Model (LLM) and a vector-indexed database, the system addresses the limitations of standard filtering through a custom "relaxed search" algorithm.
This ensures the system remains functional under complex, multi-criteria constraints while prioritizing health-specific parameters.
Data Source: https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews

Implementation Guide
1. Obtain the necessary API credentials, store them in a .env file within the root directory and install the dataset through kaggle:
SerpApi Key: https://serpapi.com/manage-api-key
Groq API Key: https://console.groq.com/keys
Dataset: https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews

3. Install the required Python packages: pip install -r requirements.txt

4. Execute the following scripts in sequence to process the raw Kaggle dataset and generate the vector store:
python processing.py
python preprocess2.py
python preprocess3.py
python build_index.py

5. To initialize the full-stack application, run the backend API followed by the frontend dashboard:
python api_app.py
streamlit run ui_app.py
