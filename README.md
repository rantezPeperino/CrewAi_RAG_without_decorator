# CrewAi RAG Router Loader Splitter Store

Proyecto multiagente con CrewAi **SIN DECORATOR** (`@crew`, `@task`, `@agent`, `@CrewBase`), para realizar RAG (Retrieval-Augmented Generation) con documentos en formatos PDF, HTML y CSV.

**No se utilizan decorator porque se agrego una secuencia de tareas para la lectura, no es posible que se leean desde los archivos de condiguracicon yaml, solo sirven para una secuencia de tareas.**

Las tareas son las que estructuran el proyecto, y los agentes se encargan de ejecutarlas, en este caso de manera secuencial con `Process.sequential`. Las tareas incluyen:

- **route_document**: Definir el tipo de documento.
- **load_document**: Cargar el documento.
- **split_document**: Dividir el documento en partes (chunks).
- **store_vectors**: Almacenar los chunks del documento en una base de datos vectorial.

### Agentes
Se crearon 4 agentes para el procesamiento:

1. **Route**: Determina el tipo de archivo (PDF, HTML o CSV).
2. **document_loader**: Encargado de cargar el documento. Usa herramientas personalizadas según el tipo de archivo, como `PDFLoaderTool`, `CSVLoaderTool` y `HTMLLoaderTool`.
3. **document_splitter**: Responsable de dividir el documento en chunks.
4. **vector_store**: Almacena los chunks en la base vectorial después del procesamiento con el LLM (Language Model).

