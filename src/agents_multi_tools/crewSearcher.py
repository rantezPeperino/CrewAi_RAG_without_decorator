# crewSearcher.py
from crewai import Agent, Crew, Process, Task
from tools.custom_tool_searcher import ChromaSearchTool, OpenAIResponseTool

class DocumentSearcherCrew:
    """Crew para búsqueda en base de datos vectorial"""
    
    def __init__(self):
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        self.crew = self._create_crew()
    
    def _create_agents(self):
        route_search = Agent(
            role="Search Router",
            goal="Analizar y dirigir la entrada de la consulta al agente correspondiente",
            backstory="Experto en identificar consultas para base de datos vectorial",
            verbose=True,
            allow_delegation=False
        )
        
        store_search = Agent(
            role="Search Store Manager",
            goal="Gestionar las búsquedas en la base de datos vectorial",
            backstory="Especialista en bases de datos vectoriales y embeddings",
            allow_delegation=False,
            tools=[ChromaSearchTool()],
            verbose=True
        )
        
        analyst_reporting = Agent(
            role="Reporting Analyst",
            goal="Analizar y reportar resultados de búsqueda",
            backstory="Especialista en análisis de datos y generación de reportes concisos",
            allow_delegation=False,
            tools=[OpenAIResponseTool()],
            verbose=True
        )
        
        return [route_search, store_search, analyst_reporting]
    
    def _create_tasks(self):
        search_route = Task(
            description="Analiza la consulta '{prompt}' para determinar la estrategia de búsqueda",
            expected_output="Estrategia de búsqueda determinada",
            agent=self.agents[0]
        )
        
        execute_search = Task(
            description="Realizar búsqueda en base de datos según la consulta '{prompt}'",
            expected_output="Resultados de la búsqueda",
            agent=self.agents[1],
            context=[search_route]
        )
        
        create_report = Task(
            description="Genera un reporte conciso con los 2 mejores resultados encontrados",
            expected_output="Reporte formateado con emojis y estructura clara",
            agent=self.agents[2],
            context=[execute_search]
        )
        
        return [search_route, execute_search, create_report]
    
    def _create_crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
    
    def search(self, prompt: str):
        return self.crew.kickoff(inputs={"prompt": prompt})