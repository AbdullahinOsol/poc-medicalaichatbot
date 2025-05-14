from langchain.docstore.document import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
import hashlib
import uuid
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()


embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("INDEX_NAME")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

MARKDOWN_SEPARATORS = [
    '\n#{1,6} ',
    '```\n',
    '\n\\*\\*\\*+\n',
    '\n---+\n',
    '\n___+\n',
    '\n\n',
    '\n',
    ' ',
    '',
]

splitter = RecursiveCharacterTextSplitter(
    separators = MARKDOWN_SEPARATORS,
    chunk_size = 650,
    chunk_overlap = 250,
)

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    multi_process=False,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

def vector_and_upsert(text):
    raw_database = LangchainDocument(page_content=text)

    if len(text) < splitter._chunk_size:
        processed_data = [raw_database]
    else:
        processed_data = splitter.split_documents([raw_database])

    data_to_add = []

    for i, entry in enumerate(processed_data):
        chunk_text = entry.page_content
        vector = embedding_model.embed_query(chunk_text)
        unique_id = str(uuid.uuid4())
        print('\n\n the unique id is: med_' + unique_id + '\n\n')

        data_to_add.append({
            'id': f'med_{unique_id}',
            'values': vector,
            'metadata': {'text': chunk_text}
        })

    return data_to_add

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

hypothesis_template = 'this sentence is about the following domain {}'
classes_verbalised = ['medical', 'non-medical']

def check_domain(query):
    zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33")
    output = zeroshot_classifier(query, classes_verbalised, hypothesis_template=hypothesis_template, multi_label=False)
    return output['labels'][0]