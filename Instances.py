from PyQt5.QtWidgets import QApplication, QMainWindow, QDialogButtonBox, QVBoxLayout, QTextEdit, QWidget, QDialog, QLineEdit, QPushButton, QScrollArea, QLabel, QSpacerItem, QSizePolicy, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox, QTabWidget,  QCheckBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
import requests
import json
import sys
import os
import uuid


# Replace with your Ollama API endpoint
api_url = "http://localhost:11434/api/chat"

class WorkerThread(QThread):
    data_received = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model, messages):
        super().__init__()
        self.model = model
        self.messages = messages

    def run(self):
        try:
            response = requests.post(api_url, json={"model": self.model, "messages": self.messages}, stream=True)
            response.raise_for_status()

            accumulated_data = ""

            for line in response.iter_lines():
                if line:
                    line_data = line.decode('utf-8')
                    if "<start_of_turn>" in line_data or "<end_of_turn>" in line_data:
                        continue
                    try:
                        response_data = json.loads(line_data)
                        if 'message' in response_data and response_data['message']['role'] == 'assistant':
                            content = response_data['message']['content']
                            accumulated_data += content
                            self.data_received.emit(content)
                            if response_data.get("done", False):
                                break
                    except json.JSONDecodeError:
                        print("Error decoding JSON:", line_data)
        except requests.exceptions.RequestException as e:
            print("Error connecting to the API:", e)
        finally:
            self.finished.emit()

class ChatBubble(QWidget):
    def __init__(self, text, is_user, is_loading=False):
        super().__init__()
        self.is_loading = is_loading

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Enable text selection
        
        if is_user:
            self.text_label.setAlignment(Qt.AlignRight)
            self.layout.addWidget(self.text_label)
            self.layout.setAlignment(Qt.AlignRight)
            self.text_label.setStyleSheet("background-color: lightgrey; color: black; padding: 10px; border-radius: 10px;")
        else:
            self.text_label.setAlignment(Qt.AlignLeft)
            self.layout.addWidget(self.text_label)
            self.layout.setAlignment(Qt.AlignLeft)
            self.text_label.setStyleSheet("background-color: white; color: black; padding: 10px; border-radius: 10px;")
        
        if is_loading:
            self.text_label.setText("...loading")
            self.text_label.setAlignment(Qt.AlignLeft)
            self.layout.setAlignment(Qt.AlignLeft)

    def append_text(self, new_text):
        current_text = self.text_label.text()
        self.text_label.setText(current_text + new_text)
        self.text_label.adjustSize()
        self.adjustSize()

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ragabond")
        self.setGeometry(100, 100, 1000, 600)
        self.setMinimumSize(600, 300)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        self.sidebar = Sidebar(self)
        self.layout.addWidget(self.sidebar)

        self.main_container = QWidget()
        self.layout.addWidget(self.main_container)
        self.main_layout = QVBoxLayout(self.main_container)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)

        self.scroll_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_widget.setObjectName("chatContainer")
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_layout.setSpacing(10)

        self.scroll_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.center_widget = QWidget()
        self.center_layout = QVBoxLayout(self.center_widget)
        self.center_layout.setAlignment(Qt.AlignCenter)
        self.scroll_layout.addWidget(self.center_widget)

        self.input_container = QWidget()
        self.input_layout = QHBoxLayout(self.input_container)
        self.input_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.addWidget(self.input_container)

        self.input_line = QLineEdit(self.input_container)
        self.input_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.input_layout.addWidget(self.input_line)

        self.send_button = QPushButton(self.input_container)
        self.send_button.setFixedSize(40, 40)
        self.send_button.setIcon(QIcon("path/to/your/icon.png"))
        self.input_layout.addWidget(self.send_button)
        self.send_button.clicked.connect(self.handle_send)
        self.send_button.setStyleSheet("background-color: black; color: white; border: none; border-radius: 5px;")


        self.input_line.returnPressed.connect(self.handle_send)

        self.load_styles()

        self.user_scrolling = False
        self.scroll_area.verticalScrollBar().valueChanged.connect(self.on_scroll)

        self.current_conversation_id = None


    def load_styles(self):
        with open("styles.qss", "r") as f:
            self.setStyleSheet(f.read())

    def handle_send(self):
        if self.current_conversation_id is None:
            print("No conversation selected.")
            return

        user_text = self.input_line.text().strip()

        self.add_chat_bubble(user_text, True)
        self.input_line.clear()
        self.sidebar.add_message_to_conversation(self.current_conversation_id, "user", user_text)

        # Add loading bubble
        self.loading_bubble = ChatBubble("", False, is_loading=True)
        self.center_layout.addWidget(self.loading_bubble)
        self.scroll_to_bottom()

        # Start worker thread if conversation ID is valid
        if self.current_conversation_id in self.sidebar.conversations:
            self.worker_thread = WorkerThread("CY.AI2", self.sidebar.conversations[self.current_conversation_id])
            self.worker_thread.data_received.connect(self.update_chat_bubble)
            self.worker_thread.finished.connect(self.on_finished)
            self.worker_thread.start()
        else:
            print(f"Conversation ID {self.current_conversation_id} not found.")

    def add_chat_bubble(self, text, is_user):
        bubble = ChatBubble(text, is_user)
        self.center_layout.addWidget(bubble)
        self.scroll_to_bottom()

    def update_chat_bubble(self, content):
        if hasattr(self, 'loading_bubble'):
            if self.loading_bubble.text_label.text() == "...loading":
                self.loading_bubble.text_label.setText("")
            self.loading_bubble.append_text(content)
            self.loading_bubble.text_label.adjustSize()  # Ensure label size is updated
            self.loading_bubble.adjustSize()  # Ensure bubble size is updated
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        if not self.user_scrolling:
            QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        scroll_bar = self.scroll_area.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def on_scroll(self, value):
        scroll_bar = self.scroll_area.verticalScrollBar()
        if scroll_bar.value() < (scroll_bar.maximum() - 1):
            self.user_scrolling = True
        else:
            self.user_scrolling = False

    def on_finished(self):
        if hasattr(self, 'loading_bubble'):
            final_content = self.loading_bubble.text_label.text()
            self.center_layout.removeWidget(self.loading_bubble)
            self.loading_bubble.deleteLater()
            del self.loading_bubble
            self.add_chat_bubble(final_content, False)
            self.sidebar.add_message_to_conversation(self.current_conversation_id, "assistant", final_content)

    def select_conversation(self, conversation_id):
        if conversation_id in self.sidebar.conversations:
            self.current_conversation_id = conversation_id
            self.center_widget.setParent(None)
            self.center_widget = QWidget()
            self.center_layout = QVBoxLayout(self.center_widget)
            self.center_layout.setAlignment(Qt.AlignCenter)
            self.scroll_layout.addWidget(self.center_widget)
            for message in self.sidebar.conversations.get(conversation_id, []):
                is_user = message["role"] == "user"
                self.add_chat_bubble(message["content"], is_user)
            self.scroll_to_bottom()
        else:
            print(f"Conversation ID {conversation_id} is not available.")


from PyQt5.QtWidgets import QFileDialog

from RAGsidebar import RAGSidebar  # Import the RAGSidebar class

class Sidebar(QWidget):
    def __init__(self, chat_window):
        super().__init__()
        self.chat_window = chat_window
        self.layout = QVBoxLayout(self)
        self.setFixedWidth(250)  # Adjust width as needed

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        # Create tabs for projects and settings
        self.projects_tab = QWidget()
        self.projects_layout = QVBoxLayout(self.projects_tab)
        self.tab_widget.addTab(self.projects_tab, "AI Multi-Session Chat Archive")

        # Replace the placeholder RAG tab with the actual RAGSidebar
        # self.rag_tab = RAGSidebar()  # Use the actual RAGSidebar class
        # self.tab_widget.addTab(self.rag_tab, "RAG")

        # Add components to Projects tab
        self.project_list = QListWidget(self)
        self.project_list.currentItemChanged.connect(self.select_conversation)
        self.projects_layout.addWidget(self.project_list)

        self.new_project_button = QPushButton("Create New Instance", self)
        self.new_project_button.setObjectName("new_project_button")
        self.projects_layout.addWidget(self.new_project_button)
        self.new_project_button.clicked.connect(self.create_new_conversation)
        self.new_project_button.setStyleSheet("background-color: black; color: white; border: none; border-radius: 5px;")

        self.conversations = {}

        self.load_conversations()

    def create_new_conversation(self):
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = []
        self.add_project_to_list(conversation_id)
        self.chat_window.select_conversation(conversation_id)

    def add_project_to_list(self, conversation_id):
        item = QListWidgetItem()
        delete_button = QPushButton()
        delete_button.setFixedSize(20, 20)
        delete_button.setIcon(QIcon("path/to/delete_icon.png"))
        delete_button.setStyleSheet("background: red; border-radius: 10px; border: none;")
        delete_button.clicked.connect(lambda: self.delete_conversation(conversation_id))
        delete_button.setObjectName("delete_button")
        
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.addWidget(QLabel(f"Instance {conversation_id[:8]}"))
        item_layout.addStretch()
        item_layout.addWidget(delete_button)
        item_widget.setLayout(item_layout)
        
        item.setSizeHint(item_widget.sizeHint())
        self.project_list.addItem(item)
        self.project_list.setItemWidget(item, item_widget)

    def delete_conversation(self, conversation_id):
        confirmation = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete Instance {conversation_id[:8]} forever?", QMessageBox.Yes | QMessageBox.No)
        if confirmation == QMessageBox.Yes:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                self.save_conversations()
                self.chat_window.select_conversation(None)
                self.project_list.clear()
                self.load_conversations()

    def add_message_to_conversation(self, conversation_id, role, content):
        if conversation_id in self.conversations:
            self.conversations[conversation_id].append({"role": role, "content": content})
            self.save_conversations()

    def select_conversation(self, current, previous):
        if current:
            conversation_id = self.get_conversation_id(current)
            self.chat_window.select_conversation(conversation_id)

    def get_conversation_id(self, item):
        conversation_id = list(self.conversations.keys())[self.project_list.row(item)]
        return conversation_id

    def save_conversations(self):
        with open("conversations.json", "w") as file:
            json.dump(self.conversations, file, indent=4)

    def load_conversations(self):
        if os.path.exists("conversations.json"):
            with open("conversations.json", "r") as file:
                self.conversations = json.load(file)
                for conversation_id in self.conversations:
                    self.add_project_to_list(conversation_id)
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())