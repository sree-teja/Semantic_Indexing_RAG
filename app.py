import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget, 
    QSizePolicy, QTabWidget, QLabel, QFrame
)

from RAGsidebar import RAGSidebar  # Ensure this path is correct
from Instances import ChatWindow  # Import ChatWindow from Instances.py

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG-Powered Semantic Indexing")

        # Initialize the restart_in_progress flag
        self.restart_in_progress = False

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a QTabWidget
        self.tab_widget = QTabWidget()

        # Create the RAG Tool tab
        self.ragabond_tab = QWidget()
        self.ragabond_layout = QHBoxLayout(self.ragabond_tab)
        self.ragabond_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Sidebar for RAG Tool
        self.sidebar = RAGSidebar(parent=self)  # Pass MainWindow as the parent
        self.sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.ragabond_layout.addWidget(self.sidebar)

        # Create a vertical layout for the chatbox and its header
        chatbox_layout = QVBoxLayout()

        # Header for the chatbox
        header = QLabel("RAG Response")
        header.setStyleSheet("font-weight: bold; font-size: 14pt;")  # Style the header
        chatbox_layout.addWidget(header)

        # Chatbox for RAG Tool
        self.chatbox = QTextEdit()
        self.chatbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chatbox.setFixedHeight(500)  # Set fixed height to make it taller
        chatbox_layout.addWidget(self.chatbox)

        # Add the chatbox layout to the RAG Tool tab
        chatbox_frame = QFrame()
        chatbox_frame.setLayout(chatbox_layout)
        self.ragabond_layout.addWidget(chatbox_frame)

        # Create the Instances tab
        self.instances_tab = QWidget()
        self.instances_layout = QVBoxLayout(self.instances_tab)
        self.instances_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Add ChatWindow to Instances tab
        self.chat_window = ChatWindow()  # Instantiate ChatWindow
        self.instances_layout.addWidget(self.chat_window)

        # Add the tabs to the tab widget
        self.tab_widget.addTab(self.ragabond_tab, "RAG Tool")
        self.tab_widget.addTab(self.instances_tab, "General Conversation")

        # Set the layout for the central widget
        central_layout = QVBoxLayout()
        central_layout.addWidget(self.tab_widget)
        central_widget.setLayout(central_layout)

        # Set initial size for the main window
        self.resize(800, 600)

        # Ensure that the layout and widgets are updated
        self.update_layout()

    def update_layout(self):
        """Force layout updates."""
        self.ragabond_tab.adjustSize()
        self.instances_tab.adjustSize()
        self.tab_widget.update()
        self.tab_widget.resize(self.tab_widget.sizeHint())
        self.update()

    def reset_state(self):
        """Reset the application state only if restarting."""
        if self.restart_in_progress:
            self.sidebar.reset()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sidebar = main_window.sidebar
    main_window.reset_state()
    sidebar.delete_index_after_restart()  # Ensure this method is defined in RAGSidebar
    main_window.show()
    sys.exit(app.exec_())
