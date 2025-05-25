from extrapolation import Extrapolation
import matplotlib.pyplot as plt
from loading import LoadingAndAggregation
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QTableView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import sys
import os


class ExtrapolationApp(QtWidgets.QWidget):
    default_folder_processed = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "processed_datasets")
    default_folder_raw = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "raw_datasets")
    filelist = ['Potatoes.csv', 'Beetroots.csv', 'Wheat_flour.csv',
                'Fish.csv', 'Carrots.csv', 'Buckwheat.csv',
                'Rice.csv', 'Meat.csv', 'Eggs.csv',
                'Milk.csv', 'Onions.csv', 'Oil.csv',
                'Sugar.csv', 'Bread.csv']

    productlist = ['Bread', 'Meat', 'Sugar', 'Fish', 'Eggs', 'Rice', 'Oil', 'Milk', 'Potatoes', 'Buckwheat', 'Carrots', 'Beetroots', 'Onions', "Wheat flour"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Extrapolation Viewer")
        self.resize(1600, 1000)
        self.layout = QtWidgets.QVBoxLayout(self)

        # Змінні для збереження поточного canvas і fig
        self.current_canvas = None
        self.current_figure = None

        # Змінна для шляху збереження агрегованих даних (за замовчуванням)
        self.selected_folder_processed = ExtrapolationApp.default_folder_processed

        # Змінна для шляху збереження сирих даних (за замовчуванням)
        self.selected_folder_raw = ExtrapolationApp.default_folder_raw

        # Кнопка для вибору папки для збереження агрегованих даних
        self.button_choose_folder_processed = QtWidgets.QPushButton("Обрати папку для збереження агрегованих даних")
        self.layout.addWidget(self.button_choose_folder_processed)
        self.button_choose_folder_processed.clicked.connect(self.choose_folder_processed)  # Виправлено помилку

        # Кнопка для вибору папки для збереження сирих даних
        self.button_choose_folder_raw = QtWidgets.QPushButton("Обрати папку для збереження сирих даних")
        self.layout.addWidget(self.button_choose_folder_raw)
        self.button_choose_folder_raw.clicked.connect(self.choose_folder_raw)  # Виправлено помилку

        # Показуємо вибрану папку для агрегованих даних
        self.label_folder_processed = QtWidgets.QLabel(f"Папка для збереження агрегованих даних: {self.selected_folder_processed}")
        self.layout.addWidget(self.label_folder_processed)

        # Показуємо вибрану папку для сирих даних
        self.label_folder_raw = QtWidgets.QLabel(f"Папка для збереження сирих даних: {self.selected_folder_raw}")
        self.layout.addWidget(self.label_folder_raw)

        # Кнопка для опрацювання та збереження даних
        self.button_loading = QtWidgets.QPushButton("Опрацювати та зберегти")
        self.layout.addWidget(self.button_loading)
        self.button_loading.clicked.connect(self.load_data)

        # Кнопка для видалення даних
        self.button_deleting = QtWidgets.QPushButton("Видалити дані")
        self.layout.addWidget(self.button_deleting)
        self.button_deleting.clicked.connect(self.delete_data)

        # Випадаючий список категорій
        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(ExtrapolationApp.productlist)
        self.layout.addWidget(self.combo_box)

        # Випадаючий список екстраполяций
        self.combo_box_extr = QtWidgets.QComboBox()
        self.combo_box_extr.addItems(["Polinomial", "Linear"])
        self.layout.addWidget(self.combo_box_extr)

        # Ідентифікатор чи додавати сезонність
        self.checkbox_example = QtWidgets.QCheckBox("Увімкнути штучну сезонність")
        self.layout.addWidget(self.checkbox_example)

        # Ідентифікатор чи додавати сезонність
        self.checkbox_moving_average = QtWidgets.QCheckBox("Увімкнути плинне середнє")
        self.layout.addWidget(self.checkbox_moving_average)

        # Поле вибору дати
        self.date_edit = QtWidgets.QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM")
        self.date_edit.setDate(QDate.currentDate())
        self.layout.addWidget(self.date_edit)

        # Кнопка для запуску побудови гістограми
        self.button_hist = QtWidgets.QPushButton("Показати гістограму")
        self.layout.addWidget(self.button_hist)
        self.button_hist.clicked.connect(self.plot_hist)

        # Кнопка для запуску побудови box-plot
        self.button_box = QtWidgets.QPushButton("Показати box-plot")
        self.layout.addWidget(self.button_box)
        self.button_box.clicked.connect(self.box_plot)

        # Кнопка для запуску побудови графіка
        self.button_eta = QtWidgets.QPushButton("Показати графік")
        self.layout.addWidget(self.button_eta)
        self.button_eta.clicked.connect(self.plot_graph)

        # Кнопка для запуску побудови heat-map
        self.button_heat_map = QtWidgets.QPushButton("Показати heat-map")
        self.layout.addWidget(self.button_heat_map)
        self.button_heat_map.clicked.connect(self.heat_map)

        # Кнопка для запуску побудови графіка екстраполяції
        self.button_extrapolation = QtWidgets.QPushButton("Показати екстраполяцію")
        self.layout.addWidget(self.button_extrapolation)
        self.button_extrapolation.clicked.connect(self.plot_extrapolation)

        # Кнопка для запуску побудови графіка екстраполяції
        self.button_validate_extrapolation = QtWidgets.QPushButton("Показати перевірку моделі екстраполяції")
        self.layout.addWidget(self.button_validate_extrapolation)
        self.button_validate_extrapolation.clicked.connect(self.validate_extrapolation)

        # Frame (QWidget) для графіка
        self.graph_container = QtWidgets.QWidget()
        self.graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        self.layout.addWidget(self.graph_container)

    def choose_folder_processed(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження агрегованих даних")
        if folder:
            self.selected_folder_processed = folder
            self.label_folder_processed.setText(f"Папка для збереження агрегованих даних: {self.selected_folder_processed}")

    def choose_folder_raw(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження сирих даних")
        if folder:
            self.selected_folder_raw = folder
            self.label_folder_raw.setText(f"Папка для збереження сирих даних: {self.selected_folder_raw}")

    def clear_graph(self):
        # Очистити попередні графіки та закрити фігуру
        while self.graph_layout.count():
            child = self.graph_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        if self.current_figure:
            plt.close(self.current_figure)  # Закриваємо попередню фігуру
            self.current_figure = None
        self.current_canvas = None

    def plot_graph(self):
        if not (os.path.isdir(self.selected_folder_processed) and os.listdir(self.selected_folder_processed) == ExtrapolationApp.filelist):
            QtWidgets.QMessageBox.information(self, "Помилка", "Дані ще не завантажені")
            return
        self.clear_graph()

        selected_category = self.combo_box.currentText()
        self.current_canvas, self.current_figure = Extrapolation.run_eta(selected_category, self.selected_folder_processed, self.graph_container)
        self.graph_layout.addWidget(self.current_canvas)

    def plot_hist(self):
        if not (os.path.isdir(self.selected_folder_processed) and os.listdir(self.selected_folder_processed) == ExtrapolationApp.filelist):
            QtWidgets.QMessageBox.information(self, "Помилка", "Дані ще не завантажені")
            return
        self.clear_graph()

        selected_category = self.combo_box.currentText()
        qdate = self.date_edit.date()
        selected_date = QDate(qdate.year(), qdate.month(), 15)
        min_date = QDate(2020, 1, 15)
        max_date = QDate(2024, 12, 15)
        if not (min_date <= selected_date <= max_date):
            QtWidgets.QMessageBox.information(self, "Помилка", "Дата має бути в межах: '2020-01 - 2024-12'")
            return
        hist_date = selected_date.toString("yyyy-MM-dd")

        self.current_canvas, self.current_figure = Extrapolation.run_hist(hist_date, selected_category, self.selected_folder_raw, self.graph_container)
        if self.current_canvas:
            self.graph_layout.addWidget(self.current_canvas)

    def box_plot(self):
        if not (os.path.isdir(self.selected_folder_processed) and os.listdir(self.selected_folder_processed) == ExtrapolationApp.filelist):
            QtWidgets.QMessageBox.information(self, "Помилка", "Дані ще не завантажені")
            return
        self.clear_graph()

        selected_category = self.combo_box.currentText()
        qdate = self.date_edit.date()
        selected_date = QDate(qdate.year(), qdate.month(), 15)
        min_date = QDate(2020, 1, 15)
        max_date = QDate(2024, 12, 15)
        if not (min_date <= selected_date <= max_date):
            QtWidgets.QMessageBox.information(self, "Помилка", "Дата має бути в межах: '2020-01 - 2024-12'")
            return
        plot_date = selected_date.toString("yyyy-MM-dd")

        self.current_canvas, self.current_figure = Extrapolation.run_box_plot(plot_date, selected_category, self.selected_folder_raw, self.graph_container)
        if self.current_canvas:
            self.graph_layout.addWidget(self.current_canvas)

    def heat_map(self):
        if not (os.path.isdir(self.selected_folder_processed) and os.listdir(self.selected_folder_processed) == ExtrapolationApp.filelist):
            QtWidgets.QMessageBox.information(self, "Помилка", "Дані ще не завантажені")
            return
        self.clear_graph()

        selected_category = self.combo_box.currentText()
        self.current_canvas, self.current_figure = Extrapolation.run_heat_map(selected_category, self.selected_folder_processed, self.graph_container)
        if self.current_canvas:
            self.graph_layout.addWidget(self.current_canvas)
        else:
            QtWidgets.QMessageBox.information(self, "Помилка", "Немає даних для побудови теплової карти")

    def plot_extrapolation(self):
        season = True
        moving_average = True
        if not self.checkbox_example.isChecked():
            season = False
        if not self.checkbox_moving_average.isChecked():
            moving_average = False
        selected_date = self.date_edit.date()
        selected_extr = self.combo_box_extr.currentText()
        min_date = QDate(2025, 1, 1)
        if selected_date < min_date:
            QtWidgets.QMessageBox.information(self, "Помилка", "Дата має бути більшою за '2024-12'")
            return
        to_extr_date = self.date_edit.date().toString("yyyy-MM-dd")
        if not (os.path.isdir(self.selected_folder_processed) and os.listdir(self.selected_folder_processed) == ExtrapolationApp.filelist):
            QtWidgets.QMessageBox.information(self, "Помилка", "Дані ще не завантажені")
            return
        self.clear_graph()

        selected_category = self.combo_box.currentText()
        self.current_canvas, self.current_figure = Extrapolation.run_extrapolation(moving_average, season, selected_extr, to_extr_date, self.selected_folder_processed, selected_category, self.graph_container)
        if self.current_canvas:  # Перевіряємо, чи повернувся canvas
            self.graph_layout.addWidget(self.current_canvas)

    def validate_extrapolation(self):
        season = self.checkbox_example.isChecked()
        moving_average = self.checkbox_moving_average.isChecked()
        selected_extr = self.combo_box_extr.currentText()

        if not (os.path.isdir(self.selected_folder_processed) and os.listdir(
                self.selected_folder_processed) == ExtrapolationApp.filelist):
            QtWidgets.QMessageBox.information(self, "Помилка", "Дані ще не завантажені")
            return

        self.clear_graph()

        selected_category = self.combo_box.currentText()

        df, mse = Extrapolation.validate_extrapolation(
            season,
            moving_average,
            selected_extr,
            self.selected_folder_processed,
            selected_category,
        )

        if df is not None and not df.empty and mse is not None:
            # Створюємо модель
            model = QStandardItemModel()
            model.setColumnCount(len(df.columns))
            model.setHorizontalHeaderLabels(df.columns.astype(str).tolist())

            for row in df.itertuples(index=False):
                items = [QStandardItem(str(cell)) for cell in row]
                model.appendRow(items)

            mse_row = []
            for col in df.columns:
                if col == 'predicted':
                    mse_row.append(QStandardItem(str(round(mse, 5))))
                elif col == 'date':
                    mse_row.append(QStandardItem("MSE"))
                else:
                    mse_row.append(QStandardItem(""))  # порожні клітинки

            model.appendRow(mse_row)

            table_view = QTableView()
            table_view.setModel(model)

            # Очистка layout перед додаванням нової таблиці
            for i in reversed(range(self.graph_layout.count())):
                widget = self.graph_layout.itemAt(i).widget()
                self.graph_layout.removeWidget(widget)
                widget.deleteLater()

            self.graph_layout.addWidget(table_view)

    def delete_data(self):
        if os.path.isdir(self.selected_folder_processed) and os.path.isdir(self.selected_folder_raw) and os.listdir(
                self.selected_folder_processed) == ExtrapolationApp.filelist and os.listdir(
                self.selected_folder_raw) == ExtrapolationApp.filelist:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Підтвердження видалення",
                "Ви впевнені, що хочете видалити всі дані?\nЦю дію неможливо скасувати.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                for filename in os.listdir(self.selected_folder_processed):
                    file_path = os.path.join(self.selected_folder_processed, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                for filename in os.listdir(self.selected_folder_raw):
                    file_path = os.path.join(self.selected_folder_raw, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                QtWidgets.QMessageBox.information(self, "Готово", "Дані успішно видалені!")
        else:
            QtWidgets.QMessageBox.information(self, "Помилка", "Дані вже видалені")

    def load_data(self):
        if os.path.isdir(self.selected_folder_processed) and os.path.isdir(self.selected_folder_raw) and os.listdir(self.selected_folder_processed) == ExtrapolationApp.filelist and os.listdir(self.selected_folder_raw) == ExtrapolationApp.filelist:
            QtWidgets.QMessageBox.information(self, "Інформація", "Дані вже завантажені!")
        else:
            LoadingAndAggregation.run_pipeline(self.selected_folder_processed, self.selected_folder_raw)
            QtWidgets.QMessageBox.information(self, "Готово", "Дані успішно оброблено та збережено!")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ExtrapolationApp()
    window.show()
    sys.exit(app.exec())