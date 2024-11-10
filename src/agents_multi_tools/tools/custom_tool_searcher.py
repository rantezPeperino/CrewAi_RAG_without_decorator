from typing import Type
from crewai_tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader 
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
import openai
from typing import Optional, Any, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
from langchain.schema import Document
import shutil
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
import json







class ChromaSearchTool(BaseTool):
    name: str = "chroma_search"
    description: str = "Busca información en la base de datos vectorial local Chroma"
    db: Any = Field(default=None)
    retriever: Any = Field(default=None)
    
    
    def __init__(self):
        super().__init__()
        model_name = "all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False} 
        cargar_modelo_embeddings = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)    

        # Cargar la base de datos existente
        persist_directory = os.path.join(os.getcwd(), "chroma_db_local")
        embedding_function = cargar_modelo_embeddings
        
        # Cargar la base de datos existente
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def _run(self, prompt: str) -> str:
        try:
            results = self.retriever.get_relevant_documents(prompt)
            contenido = "\n\n".join([doc.page_content for doc in results])
            return contenido if contenido else "No se encontraron resultados relevantes."
        except Exception as e:
            return f"Error al buscar en la base de datos: {str(e)}"


    
    
    
    
    
class OpenAIResponseTool(BaseTool):
    name: str = "OpenAIResponseTool"
    description: str = "Herramienta para generar respuestas usando OpenAI GPT-3.5-turbo"

    def _run(self, prompt: str, contexto: str) -> str:
        mensaje = f"Pregunta: {prompt}\n\nContexto:\n{contexto}\n\nRespuesta:"
        
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en proporcionar respuestas informativas."},
                {"role": "user", "content": mensaje}
            ],
            max_tokens=150
        )
        
        return respuesta['choices'][0]['message']['content'].strip()    