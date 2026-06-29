``` mermaid
flowchart TD
    A["User submits query via MCP client"] --> B["MCP Server: Route to search_all_codebases"]
    B --> C["Embed query using Ollama (nomic-embed-text)"]
    C --> D["Load master manifest table to get list of indexed repositories"]
    D --> E["For each repository in list:"]
    E --> F["Load LanceDB table for the repository"]
    F --> G["Perform cosine similarity search with embedded query"]
    G --> H["Limit results to top K per repository"]
    H --> I["Combine results from all repositories"]
    I --> J["Sort combined results by distance (ascending)"]
    J --> K["Apply global token budget safeguard"]
    K --> L["Return ranked results with metadata to user"]
```