from PyQt6.QtWidgets import QApplication
from Benchmark import ui

if __name__ == '__main__':
    app = QApplication([])
    ex = ui()
    ex.show()
    app.exec()