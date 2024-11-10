# crewLoader.py
from crewai import Agent, Crew, Process, Task
from tools.custom_tool_loader import PDFLoaderTool, HTMLLoaderTool, CSVLoaderTool

class DocumentLoaderCrew:
    """Crew para procesamiento de documentos"""
    
    def __init__(self):
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        self.crew = self._create_crew()
    
    def _create_agents(self):
        document_loader = Agent(
            role="Content Loader",
            goal="Cargar y extraer texto de los archivos PDF, HTML O CSV con precisión según corresponda",
            backstory="Especialista en extraer contenido de PDF, HTML O CSV, manteniendo la estructura del documento y asegurando la calidad del texto extraído",
            verbose=True,
            allow_delegation=False,
            tools=[PDFLoaderTool(), CSVLoaderTool(), HTMLLoaderTool()]
        )
        
        router = Agent(
            role="Router",
            goal="Analizar y dirigir la entrada dirigiendo al procesador correcto",
            backstory="Experto en identificar tipos de archivos, coordinar su procesamiento y diferenciar si es una consulta para derivar al buscador correspondiente",
            verbose=True,
            allow_delegation=False
        )
        
        document_splitter = Agent(
            role="Document Splitter",
            goal="Dividir documentos en chunks óptimos",
            backstory="Experto en segmentar documentos manteniendo coherencia contextual",
            verbose=True,
            allow_delegation=False
        )
        
        vector_store = Agent(
            role="Vector Store Manager",
            goal="Gestionar el almacenamiento vectorial de documentos",
            backstory="Especialista en bases de datos vectoriales y embeddings",
            verbose=True,
            allow_delegation=False
        )
        
        return [router, document_loader, document_splitter, vector_store]
    
    def _create_tasks(self):
        route_document = Task(
            description="Analiza la entrada para determinar el tipo de archivo en la ruta {file_path}",
            expected_output="El tipo de documento identificado",
            agent=self.agents[0]
        )
        
        load_document = Task(
            description="Carga el contenido del archivo según corresponda (PDF, HTML, CSV) ubicado en {file_path}",
            expected_output="El contenido completo del documento",
            agent=self.agents[1],
            context=[route_document]
        )
        
        split_document = Task(
            description="Divide el documento cargado en chunks manejables",
            expected_output="Los chunks generados del documento",
            agent=self.agents[2],
            context=[load_document]
        )
        
        store_vectors = Task(
            description="Almacena los chunks procesados en la base de datos vectorial",
            expected_output="Confirmación de almacenamiento exitoso",
            agent=self.agents[3],
            context=[split_document]
        )
        
        return [route_document, load_document, split_document, store_vectors]
    
    def _create_crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
    
    def process_file(self, file_path: str):
        return self.crew.kickoff(inputs={"file_path": file_path})