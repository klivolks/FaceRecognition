from PyQt6.QtWidgets import QApplication
from ui import CameraSelector

if __name__ == '__main__':
    app = QApplication([])
    ex = CameraSelector()
    ex.show()
    app.exec()
