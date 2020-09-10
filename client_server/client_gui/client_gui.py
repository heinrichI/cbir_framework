from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

from client_server.client_gui.qt_designer import design
from client_server.json_socket.jsocket_base import JsonClient
from core.data_store.file_system_directory_datastore import FileSystemDirectoryDataStore
from core.common.file_utils import filter_by_image_extensions

cached_pixmaps = QPixmapCache()
# print(cached_pixmaps.cacheLimit())
cached_pixmaps.setCacheLimit(100 * 2 ** 10)

DISPLAY_IMG_SIZE=200

class SimpleListModel(QAbstractListModel):
    dataChangedSignal = pyqtSignal(int, int)

    def __init__(self, mlist):
        QAbstractListModel.__init__(self)
        self._items = mlist

    def rowCount(self, parent=QModelIndex()):
        return len(self._items)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return QVariant(self._items[index.row()])
        elif role == Qt.EditRole:
            return QVariant(self._items[index.row()])
        elif role == Qt.ToolTipRole:
            return QVariant(self._items[index.row()])
        elif role == Qt.DecorationRole:
            full_path = self._items[index.row()]
            pixmap = cached_pixmaps.find(full_path)
            if not pixmap:
                pixmap = QPixmap(full_path)
                pixmap = pixmap.scaled(DISPLAY_IMG_SIZE, DISPLAY_IMG_SIZE)
                cached_pixmaps.insert(full_path, pixmap)

            return pixmap
        else:
            return QVariant()

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled


class SimpleListView(QListView):
    def __init__(self, parent=None):
        QListView.__init__(self, parent)
        self.setAlternatingRowColors(True)


def onDragDropAction(self, e):
    e.accept()
    print(e)


class MyMainWindow(QMainWindow, design.Ui_main_window):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.search_results_list_view.setGridSize(QSize(DISPLAY_IMG_SIZE, DISPLAY_IMG_SIZE))

        # self.query_images_list_view.height()
        self.query_images_list_view.clicked.connect(self.onClickAction)
        self.updateQueryListView([])

        self.setAcceptDrops(True)

        self.search_results = []
        self.updateSearchResultsView([])

        self.choose_images_btn.clicked.connect(self.onFileDialogAction)
        self.search_btn.clicked.connect(self.onSearchAction)

    def dragEnterEvent(self, e):
        # print("drag", e.mimeData().hasText())
        if e.mimeData().hasText():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        uri_list_text = e.mimeData().text()
        uris = uri_list_text.split('\n')
        img_uris = list(filter(filter_by_image_extensions, uris))
        imf_filepathes = []
        for img_uri in img_uris:
            try:
                img_filepath = img_uri.split('file:///')[1]
                imf_filepathes.append(img_filepath)
            except e:
                print(e)
        # print("drop", len(e.mimeData().text()), e.mimeData().text())
        self.updateQueryListView(imf_filepathes)
        self.updateSearchResultsView([])

    @pyqtSlot()
    def onFileDialogAction(self):
        filepathes, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                     "Image files (*.jpg *.jpeg *.png)")

        # selected_dir = r'C:\Users\User\GoogleDisk\Pictures\Mountains'
        # selected_dir = r'D:\datasets\brodatz\data.brodatz\size_213x213'
        self.updateQueryListView(filepathes)
        self.updateSearchResultsView([])

    @pyqtSlot()
    def onSearchAction(self):
        n_nearest = int(self.n_nearest.text())
        query_images_filepathes = self.query_images_model._items
        if not query_images_filepathes:
            return

        # self.search_results = fake_search_results = [[query_image_filepath] * n_nearest for query_image_filepath in query_image_filepathes]
        self.search_results = self.retrieve_search_results(query_images_filepathes, n_nearest)
        self.updateSearchResultsView(self.search_results[0])

    @pyqtSlot("QModelIndex")
    def onClickAction(self, model_index):
        if self.search_results:
            search_results_for_row = self.search_results[model_index.row()]
            self.updateSearchResultsView(search_results_for_row)

    def updateQueryListView(self, img_pathes):
        self.query_images_model = SimpleListModel(img_pathes)
        self.query_images_list_view.setModel(self.query_images_model)

    def updateSearchResultsView(self, img_pathes):
        self.result_images_model = SimpleListModel(img_pathes)
        self.search_results_list_view.setModel(self.result_images_model)

    def retrieve_search_results(self, query_images_filepathes, n_nearest):
        client = JsonClient()
        client.connect()
        query_msg = {
            'query_image_filepathes': query_images_filepathes,
            'n_nearest': n_nearest
        }
        client.send_obj(query_msg)

        nearest_neighbor_native_ids_list_of_lists = client.read_obj()
        # print(len(nearest_neighbor_native_ids_list))
        # print(nearest_neighbor_native_ids_list)
        try:
            client.send_obj(True)
            client.close()
        except:
            raise Warning("Server shutdown")

        return nearest_neighbor_native_ids_list_of_lists


def main():
    app = QApplication(sys.argv)
    w = MyMainWindow()
    w.show()
    app.exec_()


if __name__ == '__main__':
    main()
