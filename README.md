# RAG-Powered-Semantic-Indexing

This program leverages Retrieval-Augmented Generation (RAG) to enable efficient querying of large documents and databases with instant load times once the index is generated.

- **Create vector stores** from your provided data.
- **Select from a list** of your created indexes and query them semantically for direct answers.

## What is RAG?

"Retrieval-Augmented Generation (RAG) optimizes the output of a large language model by referencing an authoritative knowledge base outside of its training data before generating a response. This approach enhances the model's capabilities without retraining, ensuring the output remains relevant and accurate across various contexts." - Paraphrased from Amazon Web Services

## Additional Features

The "Instances" tab provides an interface for Local LLMs, retaining history with a Multi-Session Chat Archive. This allows users to ask more general questions and test out various local LLMs.

## Instructions

### Step 1: Setting Up Ollama

- **Download Ollama:** [Download Ollama here](https://ollama.com/).
- **Setup Guide:** Watch [this video](https://www.youtube.com/watch?v=oI7VoTM9NKQ) for setup instructions specific to your system.
- **Pull Llama3.1 and nomic-embed-text:** These are the default models used by this application. They can be replaced later by navigating to the "Instances" and "RAGsidebar" Python scripts and replacing their titles with your own models.
    ```bash
    ollama run llama3:8b
    ```
    ```bash
    ollama pull nomic-embed-text
    ```

### Step 2: Place All Files Into One Directory

- **Download All Listed Files from the Repository:** Open your directory in VS Code or navigate through the console/terminal. Then run the respective commands for your operating system listed below in Step 5.

### Step 3: Ensure Dependencies

- **Create Virtual Environment:**
    ```bash
    python -m venv venv
    ```
- **Activate Virtual Environment:**
    - **Windows:**
        ```bash
        venv\Scripts\activate
        ```
    - **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
- **Install Packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Step 4: Run `app.py`

- **Congrats on Setting Up Your New RAG Application!** You're ready to start querying. The following steps are optional and only needed if you wish to customize the AI model's personality in the general "Instances" tab.

### Step 5 (Optional): Change the Personality of Your Local LLM Chats in Instances

- **Customize YAML:** - Navigate to the customllama3.1.yaml file. The "SYSTEM" portion dictates the AI’s behavior and personality. For example, you can specify: "YOU are named John, YOU are a chef." Update the SYSTEM message with desired attributes, then run the following command in Step 7:

### Step 6 (Optional): Creating the New Model

- **Default Model Name:** The script uses `llama3.1` by default in `Instances.py`.
    ```bash
    ollama create customllama3.1 -f (YOURMODELNAMEHERE)
    ```
- **Custom Model:** To use a custom YAML file:
    ```bash
    ollama create (NAME_YOUR_MODEL) -f (NAME_OF_YOUR_CUSTOM_YAML_FILE)
    ```
- **Verify Model Creation:**
    ```bash
    ollama list
    ```

**Note:** Update the model name in `Instances.py` from "llama3.1" to your custom name. You can create multiple models with Ollama.

**Happy querying!** If you have any questions or need assistance, feel free to reach out.
