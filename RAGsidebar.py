import sys
import os
import json
import shutil
import subprocess
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget, QLabel, QListWidgetItem, QHBoxLayout, QRadioButton, QSplitter,
    QLineEdit, QPushButton, QFileDialog, QListWidget, QMessageBox, QCheckBox, QSizePolicy, QDialog, QDialogButtonBox, QTextEdit
)
from PyQt5.QtCore import Qt, QSize, QVariant, pyqtSignal, QObject, QRunnable, QThreadPool
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QDragMoveEvent
from PyPDF2 import PdfReader
import csv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

# Initialize the local LLM
llm_local = OllamaLLM(model="llama3.1")

# Function to process files
def process_pdf(file_path):
    pdf = PdfReader(file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    return pdf_text

def process_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def process_csv(file_path):
    csv_text = ""
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            csv_text += " ".join(row) + "\n"
    return csv_text

def process_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def process_files(file_paths):
    combined_text = ""
    for file_path in file_paths:
        if file_path.lower().endswith(".pdf"):
            combined_text += process_pdf(file_path)
        elif file_path.lower().endswith(".txt"):
            combined_text += process_txt(file_path)
        elif file_path.lower().endswith(".csv"):
            combined_text += process_csv(file_path)
        elif file_path.lower().endswith(".json"):
            combined_text += process_json(file_path)
        combined_text += "\n\n"
    return combined_text

def create_vector_store(text, index_path, file_paths):
    try:
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
        pbar = tqdm(total=len(texts), desc="Creating vector store", unit="chunk")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        for i, chunk in enumerate(texts):
            docsearch = Chroma.from_texts([chunk], embeddings, metadatas=[metadatas[i]], persist_directory=index_path)
            pbar.update(1)
            elapsed_time = time.time() - start_time
            time_per_chunk = elapsed_time / (i + 1)
            estimated_total_time = time_per_chunk * len(texts)
            time_remaining = estimated_total_time - elapsed_time
            pbar.set_postfix(time_remaining=f"{time_remaining:.2f}s")
        with open(os.path.join(index_path, "file_paths.json"), "w") as f:
            json.dump(file_paths, f)
        pbar.close()
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise

def load_vector_store(index_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma(persist_directory=index_path, embedding_function=embeddings)
    file_paths_json = os.path.join(index_path, "file_paths.json")
    if os.path.exists(file_paths_json):
        with open(file_paths_json, "r") as f:
            file_paths = json.load(f)
            print("The following files were used to create this semantic index:")
            for file_path in file_paths:
                print(f"- {file_path}")
    else:
        print("No record of files found for this index.")
    return docsearch

def create_conversational_chain(docsearch):
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", chat_memory=message_history, return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    return chain

def handle_tasks(query):
    """
    Detect and handle specific tasks from the query.
    """
    # Example task: listing files in a directory
    if "list files in directory" in query.lower():
        directory = query.split("directory")[-1].strip()
        if os.path.exists(directory):
            files = os.listdir(directory)
            return f"Files in {directory}:\n" + "\n".join(files)
        else:
            return f"Directory {directory} does not exist."
    
    # Example task: create a new file
    if "create a file named" in query.lower():
        parts = query.split("create a file named")
        if len(parts) > 1:
            filename = parts[-1].strip()
            try:
                with open(filename, 'w') as f:
                    f.write("File created.")
                return f"File '{filename}' created successfully."
            except Exception as e:
                return f"Failed to create file '{filename}': {e}"

    # More tasks can be added here

    return None  # If no task is detected, return None
def query_chain(chain, query, chatbox):
    try:
        task_result = handle_tasks(query)
        if task_result:
            chatbox.append(f"Task Result: {task_result}\n")
            return

        modified_query = f"""Answer the following question:\n\n{query}\n\nProvide a direct and accurate response based on the information available."""
        res = chain.invoke({"question": modified_query})
        answer = res.get("answer", "")
        source_documents = res.get("source_documents", [])

        chatbox.append(f"Query: {query}\n")
        chatbox.append(f"Answer: {answer}\n")

        if source_documents:
            for idx, doc in enumerate(source_documents):
                chatbox.append(f"Source {idx + 1}: {doc.page_content[:200]}...\n")

        chain.memory.clear()  # Wipe chat memory after each query
    except Exception as e:
        chatbox.append(f"Error querying chain: {e}\n")
        print(f"Error querying chain: {e}")

class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        result = self.fn(*self.args, **self.kwargs)
        self.signals.result.emit(result)
        self.signals.finished.emit()

class FileListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        for url in urls:
            file_path = url.toLocalFile()
            self.addItem(file_path)
        event.acceptProposedAction()

from PyQt5.QtWidgets import QListWidgetItem, QHBoxLayout, QWidget, QPushButton, QLabel, QVBoxLayout, QLineEdit, QFileDialog, QMessageBox, QTabWidget, QListWidget

class RAGSidebar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.init_ui()
        self.threadpool = QThreadPool()

    def delete_index(self, index_name):
        """Handle the delete button click by restarting the application."""
        confirm = QMessageBox.question(
            None,
            "Confirm Deletion",
            f"Are you sure you want to delete the index '{index_name}'? \n"
    "    * THE APPLICATION WILL RESTART *",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            # Save the index name to be deleted
            self.save_delete_index(index_name)

            # Restart the application
            self.restart_app()

    def save_delete_index(self, index_name):
        """Save the index name to a temporary file for deletion after restart."""
        with open("delete_index_temp.json", "w") as f:
            json.dump({"index_name": index_name}, f)

    def load_delete_index(self):
        """Load the index name from the temporary file."""
        if os.path.exists("delete_index_temp.json"):
            with open("delete_index_temp.json", "r") as f:
                data = json.load(f)
            os.remove("delete_index_temp.json")  # Clean up the temp file
            return data.get("index_name")
        return None

    def delete_index_after_restart(self):
        """Delete the index that was saved before restart."""
        index_name = self.load_delete_index()
        if index_name:
            index_path = os.path.join("chroma_indexes", index_name)
            try:
                shutil.rmtree(index_path)
                QMessageBox.information(None, "Success", f"Index '{index_name}' deleted successfully.")
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to delete index '{index_name}': {e}")


    def reset(self):
        """Reset the state of the sidebar."""
        self.index_name_input.clear()
        self.file_list.clear()
        self.query_input.clear()
        self.index_list.clear()
        self.status_label.setText("")
        self.docsearch = None

    def restart_app(self):
        """Restart the application by re-launching it."""
        # Set the flag before restarting
        self.parent_widget.restart_in_progress = True
        
        QApplication.instance().quit()
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit()
    
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.setFixedWidth(250)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        self.setup_rag_tab()

    def setup_rag_tab(self):
        self.rag_tab = QWidget()
        self.rag_layout = QVBoxLayout(self.rag_tab)
        self.tab_widget.addTab(self.rag_tab, "Create an Index to Query Directly")
        
    

        self.index_name_input = QLineEdit()
        self.index_name_input.setPlaceholderText("Enter new index name")
        self.rag_layout.addWidget(self.index_name_input)
        
        self.browse_button = QPushButton("Browse Files (Or Drag To Box Bellow)")
        self.browse_button.clicked.connect(self.browse_files)
        self.rag_layout.addWidget(self.browse_button)
        
        self.file_list = FileListWidget()
        self.rag_layout.addWidget(self.file_list)
        
        self.create_index_button = QPushButton("Create Index")
        self.create_index_button.clicked.connect(self.create_index)
        self.rag_layout.addWidget(self.create_index_button)
        
        self.index_list = QListWidget()
        self.load_existing_indexes()
        self.rag_layout.addWidget(self.index_list)
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your query")
        self.rag_layout.addWidget(self.query_input)
        
        self.query_button = QPushButton("Query Index")
        self.query_button.clicked.connect(self.query_index)
        self.rag_layout.addWidget(self.query_button)
        
        self.status_label = QLabel("")
        self.rag_layout.addWidget(self.status_label)

        self.docsearch = None

    def load_existing_indexes(self):
        self.index_list.clear()
        if os.path.exists("chroma_indexes"):
            indexes = os.listdir("chroma_indexes")
            for index in indexes:
                item = QListWidgetItem()
                item.setSizeHint(QSize(300, 30))
                item_widget = QWidget()
                layout = QHBoxLayout(item_widget)
                layout.setContentsMargins(5, 0, 5, 0)

                truncated_name = index[:20] + "..." if len(index) > 20 else index
                index_label = QLabel(truncated_name)
                index_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

                info_button = QPushButton("i")
                info_button.setFixedSize(20, 20)
                info_button.setStyleSheet("QPushButton { border-radius: 10px; background-color: lightgray; }")
                info_button.clicked.connect(lambda _, i=index: self.show_index_info(i))

                layout.addWidget(index_label)
                layout.addWidget(info_button)
                layout.addStretch()

                item_widget.setLayout(layout)
                item.setData(Qt.UserRole, QVariant(index))
                self.index_list.addItem(item)
                self.index_list.setItemWidget(item, item_widget)

            max_width = max(self.index_list.sizeHintForColumn(0) + 20, 300)
            self.index_list.setMinimumWidth(max_width)
        else:
            os.makedirs("chroma_indexes")



    def browse_files(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            for file_path in file_paths:
                self.file_list.addItem(file_path)

    def create_index(self):
        index_name = self.index_name_input.text().strip()
        if not index_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the new index.")
            return

        if not os.path.exists("chroma_indexes"):
            os.makedirs("chroma_indexes")

        index_path = os.path.join("chroma_indexes", index_name)
        if os.path.exists(index_path):
            QMessageBox.warning(self, "Error", f"Index '{index_name}' already exists.")
            return

        file_paths = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not file_paths:
            QMessageBox.warning(self, "Error", "Please select at least one file to create the index.")
            return

        os.makedirs(index_path)

    # Create a worker to handle the long-running task
        worker = Worker(self._create_index_worker, index_name, index_path, file_paths)
        worker.signals.finished.connect(lambda: self._on_index_created(index_name))
        self.threadpool.start(worker)

    def _on_index_created(self, index_name):
        self.load_existing_indexes()
        QMessageBox.information(self, "Success", f"Index '{index_name}' created successfully.\n"
    "    * THE APPLICATION WILL RESTART *"
)
        self.file_list.clear()
        self.index_name_input.clear()
        self.restart_app()

    def _create_index_worker(self, index_name, index_path, file_paths):
        combined_text = process_files(file_paths)
        create_vector_store(combined_text, index_path, file_paths)
        return index_name


    def query_index(self):
        selected_item = self.index_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Error", "Please select an index to query.")
            return

        query = self.query_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Error", "Please enter a query.")
            return

        index_name = selected_item.data(Qt.UserRole)
        index_path = os.path.join("chroma_indexes", index_name)

        if not os.path.exists(index_path):
            QMessageBox.warning(self, "Error", f"Index path '{index_path}' does not exist.")
            return

    # Load the index if it's not already loaded
        if self.docsearch is None or not self._is_same_index_loaded(index_path):
            self.docsearch = load_vector_store(index_path)
            self.current_index_path = index_path  # Store the current index path

    # Clear the chatbox before displaying the new query and its result
        self.parent_widget.chatbox.clear()

        self.wipe_memory_after_query = True

    # Create a worker for the query
        worker = Worker(self._query_index_worker, query, index_name)
        worker.signals.result.connect(lambda res: self.parent_widget.chatbox.append(res))
        worker.signals.finished.connect(lambda: self.status_label.setText(f"Query completed on index '{index_name}'"))
        self.threadpool.start(worker)



    def _is_same_index_loaded(self, index_path):
        return hasattr(self, 'current_index_path') and self.current_index_path == index_path

    def _query_index_worker(self, query, index_name):
        try:
            chain = create_conversational_chain(self.docsearch)
            return query_chain(chain, query, self.parent_widget.chatbox)
        except Exception as e:
            print(f"Error in query index worker: {e}")
            return f"Error: {e}"

    def show_index_info(self, index_name):
        index_path = os.path.join("chroma_indexes", index_name)
        file_paths_json = os.path.join(index_path, "file_paths.json")

        if not os.path.exists(file_paths_json):
            QMessageBox.warning(self, "Error", f"No file information available for index '{index_name}'.")
            return

        with open(file_paths_json, "r") as f:
            file_paths = json.load(f)

    # Create a dialog for displaying the info
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Index Info: {index_name}")
        dialog.setMinimumSize(400, 300)

        layout = QVBoxLayout(dialog)

    # Text area to display the list of files
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setText("The following files were used to create this semantic index:\n" + "\n".join([f"- {fp}" for fp in file_paths]))
        layout.addWidget(info_text)

    # Delete button
        delete_button = QPushButton("Delete Index")
        delete_button.clicked.connect(lambda: self.delete_index(index_name))
        layout.addWidget(delete_button)

    # Dialog buttons
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        dialog_buttons.accepted.connect(dialog.accept)
        layout.addWidget(dialog_buttons)

        dialog.exec_()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Application")
        
        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Sidebar
        self.sidebar = RAGSidebar(parent=self)  # Pass MainWindow as the parent
        main_layout.addWidget(self.sidebar)
        
        # Chatbox
        self.chatbox = QTextEdit()
        self.chatbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.chatbox)
        
        # Set the layout for the central widget
        central_widget.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sidebar = main_window.sidebar
    sidebar.delete_index_after_restart()
    main_window.show()
    sys.exit(app.exec_())


# add a thing that shows how much time is left/% !!!