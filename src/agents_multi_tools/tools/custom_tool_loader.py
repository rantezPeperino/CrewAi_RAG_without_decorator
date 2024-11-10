from typing import Type
from crewai_tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader 
from langchain_huggingface import HuggingFaceEmbeddings
import os
import openai
from typing import Optional, Any, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
from langchain.schema import Document
import shutil
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
import json





class PDFLoaderInput(BaseModel):
    file_path: str = Field(..., description="Ruta al archivo PDF a cargar")

class PDFLoaderTool(BaseTool):
    name: str = "PDF Loader"
    description: str = "Carga y extrae contenido de archivos PDF"
    args_schema: type[BaseModel] = PDFLoaderInput

    def _run(self, file_path: str) -> str:
        """Implementación del método run para cargar PDF"""
        try:
            # Normalizar la ruta y verificar que existe
            file_path = os.path.abspath(os.path.normpath(file_path))
            
            if not os.path.exists(file_path):
                return f"Error: El archivo no existe en la ruta: {file_path}"
            
            if not file_path.lower().endswith('.pdf'):
                return f"Error: El archivo no es un PDF: {file_path}"
                
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            if not pages:
                return "Error: El PDF está vacío o no se pudo leer contenido"
                
            text = "\n".join([page.page_content for page in pages])
            
            # Verificar que obtuvimos contenido
            if not text.strip():
                return "Error: No se pudo extraer texto del PDF"
                
            return text
            
        except Exception as e:
            return f"Error al cargar PDF: {str(e)}\nRuta del archivo: {file_path}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Implementación del método arun para versión asíncrona"""
        raise NotImplementedError("Método asíncrono no implementado")


class HTMLLoaderInput(BaseModel):
    file_path: str = Field(..., description="Ruta al archivo HTML a cargar")

class HTMLLoaderTool(BaseTool):
    name: str = "HTML Loader"
    description: str = "Carga y extrae contenido de archivos HTML"
    args_schema: type[BaseModel] = HTMLLoaderInput

    def _run(self, file_path: str) -> str:
        """Implementación del método run para cargar HTML"""
        try:
            # Normalizar la ruta y verificar que existe
            file_path = os.path.abspath(os.path.normpath(file_path))
            
            if not os.path.exists(file_path):
                return f"Error: El archivo no existe en la ruta: {file_path}"
            
            if not file_path.lower().endswith('.html'):
                return f"Error: El archivo no es un HTML: {file_path}"
                
            loader = UnstructuredHTMLLoader(file_path)
            pages = loader.load()
            
            if not pages:
                return "Error: El HTML está vacío o no se pudo leer contenido"
                
            text = "\n".join([page.page_content for page in pages])
            
            # Verificar que obtuvimos contenido
            if not text.strip():
                return "Error: No se pudo extraer texto del HTML"
                
            return text
            
        except Exception as e:
            return f"Error al cargar HTML: {str(e)}\nRuta del archivo: {file_path}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Implementación del método arun para versión asíncrona"""
        raise NotImplementedError("Método asíncrono no implementado")
        
        
        
class CSVLoaderInput(BaseModel):
    file_path: str = Field(..., description="Ruta al archivo CSV a cargar")

class CSVLoaderTool(BaseTool):
    name: str = "CSV Loader"
    description: str = "Carga y extrae contenido de archivos CSV"
    args_schema: type[BaseModel] = CSVLoaderInput

    def _run(self, file_path: str) -> str:
        """Implementación del método run para cargar CSV"""        
        try:
            # Normalizar la ruta y verificar que existe
            file_path = os.path.abspath(os.path.normpath(file_path))
            
            if not os.path.exists(file_path):
                return f"Error: El archivo no existe en la ruta: {file_path}"
            
            if not file_path.lower().endswith('.csv'):
                return f"Error: El archivo no es un CSV: {file_path}"
                
            loader = CSVLoader(file_path)
            pages = loader.load()
            
            if not pages:
                return "Error: El CSV está vacío o no se pudo leer contenido"
                
            text = "\n".join([page.page_content for page in pages])
            
            # Verificar que obtuvimos contenido
            if not text.strip():
                return "Error: No se pudo extraer texto del CSV"
                
            return text
            
        except Exception as e:
            return f"Error al cargar CVS: {str(e)}\nRuta del archivo: {file_path}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Implementación del método arun para versión asíncrona"""
        raise NotImplementedError("Método asíncrono no implementado")        



class DocumentSplitterTool(BaseTool):
    name: str = "DocumentSplitterTool"
    description: str = "Divide el texto en chunks para su procesamiento"

    def _run(self, text: str) -> list:
        try:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=300,
                chunk_overlap=50
            )
            # Crear documento y dividirlo
            doc = Document(page_content=text)
            chunks = splitter.split_documents([doc])
            # Devolver la lista de chunks
            return chunks
        except Exception as e:
            return f"Error al dividir documento: {str(e)}"
        
        
        
        


class VectorDBTool(BaseTool):
    name: str = "VectorDBTool"
    description: str = "Almacena documentos en una base de datos vectorial"

    def _run(self, text_chunks: str) -> str:
        try:
            # Convertir los chunks de texto en objetos Document
            docs = [Document(page_content=chunk) for chunk in text_chunks.split('\n')]
            
            # Crear embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )

            # Preparar directorio
            persist_directory = os.path.join(os.getcwd(), "chroma_db_local")
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)

            # Crear base de datos vectorial
            vectorstore = Chroma.from_documents(
                documents=docs,  # Ahora son objetos Document
                embedding=embeddings,
                persist_directory=persist_directory
            )

            return f"Base de datos vectorial creada con {len(docs)} documentos"
        except Exception as e:
            return f"Error al crear base de datos vectorial: {str(e)}"

class ChromaSearchTool(BaseTool):
    name: str = "chroma_search"
    description: str = "Busca información en la base de datos vectorial local Chroma"
    db: Chroma = Field(default=None)
    retriever: any = Field(default=None)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
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
