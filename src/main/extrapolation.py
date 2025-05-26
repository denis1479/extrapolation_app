import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Extrapolation:
    product_names = ['Bread', 'Meat', 'Sugar', 'Fish', 'Eggs', 'Rice', 'Oil', 'Milk', 'Potatoes', 'Buckwheat', 'Carrots', 'Beetroots', 'Onions', 'Wheat flour']

    def __new__(cls, *args, **kwargs):
        raise TypeError("це статичний клас")

    @staticmethod
    def __loading(prd_name, path):
        df = pd.read_csv(f"{path}/{prd_name.replace(' ', '_')}.csv")
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        return df

    @staticmethod
    def run_eta(prd_name, path, qt_parent):
        to_eta = Extrapolation.__loading(prd_name, path)

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(to_eta['date'], to_eta['usdprice'], marker='o', linestyle='-', color='blue')
        ax.set_title('USD Price Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('USD Price')
        ax.grid(True)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setParent(qt_parent)
        return canvas, fig

    @staticmethod
    def run_box_plot(box_plot_date, prd_name, path, qt_parent):
        to_plot = Extrapolation.__loading(prd_name, path)

        fig, ax = plt.subplots(figsize=(8, 3))

        # Фільтруємо дані за датою
        to_plot = to_plot[to_plot['date'] == box_plot_date]

        # Створюємо box plot
        ax.boxplot(to_plot['usdprice'].dropna(), vert=True, patch_artist=True, showmeans=True,
                   boxprops=dict(facecolor='lightblue', color='blue'),
                   medianprops=dict(color='red'),
                   meanprops=dict(color='green'),
                   whiskerprops=dict(color='blue'),
                   capprops=dict(color='blue'),
                   flierprops=dict(markerfacecolor='red', marker='o', markersize=5))

        ax.set_title(f'Box Plot of USD Prices {prd_name}')
        ax.set_ylabel('USD Price')
        ax.grid(True, axis='y')

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setParent(qt_parent)
        return canvas, fig

    @staticmethod
    def run_hist(hist_date, prd_name, path, qt_parent):
        to_hist = Extrapolation.__loading(prd_name, path)

        fig, ax = plt.subplots(figsize=(12, 3))

        to_hist = to_hist[to_hist['date'] == hist_date]

        prices = to_hist['usdprice'].tolist()

        # Кількість елементів
        n = len(prices)

        # 1. Середнє (mean)
        mean = sum(prices) / n

        # 2. Медіана (median)
        sorted_prices = sorted(prices)
        if n % 2 == 1:
            median = sorted_prices[n // 2]
        else:
            median = (sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) / 2

        # 3. Дисперсія (variance)
        # Варіант з поділом на n-1 (вибіркова дисперсія)
        dispersion = sum((x - mean) ** 2 for x in prices) / (n - 1)

        # 4. Стандартне відхилення (std)
        standart_deviation= dispersion ** 0.5

        # 5. Мінімальне значення (min)
        min_val = min(prices)

        # 6. Максимальне значення (max)
        max_val = max(prices)

        ax.hist(to_hist['usdprice'], bins=8, color='blue', edgecolor='black')
        ax.text(
            0.99, 0.95,
            f"Середнє: {mean}. Медіана: {median}",
            transform=ax.transAxes,
            ha='right',
            va='top',
            color='black',
            fontsize=12
        )
        ax.text(
            0.99, 0.85,
            f"Дисперсія: {dispersion}. Стандартне відхилення: {standart_deviation}",
            transform=ax.transAxes,
            ha='right',
            va='top',
            color='black',
            fontsize=12
        )
        ax.text(
            0.99, 0.75,
            f"Мінімальне значення:{min_val}. Максимальне значення: {max_val}",
            transform=ax.transAxes,
            ha='right',
            va='top',
            color='black',
            fontsize=12
        )

        ax.set_title('Histogram of USD Prices')
        ax.set_xlabel('USD Price')
        ax.set_ylabel('Frequency')
        ax.grid(True, axis='y')

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setParent(qt_parent)
        return canvas, fig

    @staticmethod
    def run_heat_map(prd_name, path, qt_parent):
        to_eta = Extrapolation.__loading(prd_name, path)

        # Перевіряємо, чи є дати
        if to_eta.empty or 'date' not in to_eta.columns:
            return None, None

        # Додаємо рік і місяць
        to_eta['year'] = to_eta['date'].dt.year
        to_eta['month'] = to_eta['date'].dt.month

        # Pivot table: рік — рядки, місяць — стовпці
        pivot_table = to_eta.pivot_table(
            index='year',
            columns='month',
            values='usdprice',
            aggfunc='mean'
        )

        # Побудова графіка
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

        ax.set_title(f'Average {prd_name} Price Heatmap (Year x Month)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        ax.tick_params(axis='y', labelsize=8)
        fig.tight_layout()

        # PyQt-віджет
        canvas = FigureCanvas(fig)
        canvas.setParent(qt_parent)
        return canvas, fig

    @staticmethod
    def predict_with_date(col_name, season, last_row, x_start_date, x_end_date, start_year, start_month, a, b, c):
        amp = 0.05
        freq = 2 * np.pi / 12
        phase = 0

        last_row = last_row.rename(columns={col_name: 'predicted_usdprice'})
        predicted_df = pd.DataFrame(columns=['date', 'predicted_usdprice'])
        if predicted_df.empty:
            predicted_df = last_row
        else:
            predicted_df = pd.concat([last_row, predicted_df], ignore_index=True)

        x_start_date = x_start_date + pd.DateOffset(months=1)
        start_date = x_start_date.replace(day=15)
        x_start_index = ((start_date.year - start_year) * 12) + (start_date.month - start_month)

        end_date = pd.to_datetime(x_end_date, format='%Y-%m-%d')
        end_date = end_date.replace(day=15)
        x_end_index = ((end_date.year - start_year) * 12) + (end_date.month - start_month)

        period = 0
        for x_index in range(x_start_index, x_end_index+1):
            if a is None:
                y_pred = b * x_index + c
            else:
                y_pred = a * x_index**2 + b * x_index + c
            if season:
                season = amp * np.sin(freq * x_index + phase)
                trend = y_pred + season
            else:
                trend = y_pred
            pred_date = start_date + pd.DateOffset(months=period)
            new_row = pd.DataFrame({
                'date': pred_date,
                'predicted_usdprice': [trend]
            })

            if predicted_df.empty:
                predicted_df = new_row
            else:
                predicted_df = pd.concat([predicted_df, new_row], ignore_index=True)
            period += 1
        predicted_df['inflation_rate'] = predicted_df['predicted_usdprice'].pct_change()
        first_value = predicted_df.iloc[0]['predicted_usdprice']
        last_value = predicted_df.iloc[-1]['predicted_usdprice']
        inflation_over_all_period = ((last_value-first_value)/first_value)*100
        predicted_df = predicted_df.drop(predicted_df.index[0]).reset_index(drop=True)
        return predicted_df, inflation_over_all_period

    @staticmethod
    def gausian(x):
        if not isinstance(x, list):
            return x
        for i in range(0, len(x)-1):
            if x[i][i] == 0:
                raise ValueError(f"Ділення на нуль виявлено при спробі поділити на {i}|{i}-ий елемент")
            for j in range(i+1, len(x)):
                m = x[j][i] / x[i][i]
                for l in range(len(x[i])):
                    x[j][l] -= x[i][l]*(m)
        for i in range(len(x)-1, 0, -1):
            if x[i][i] == 0:
                raise ValueError(f"Ділення на нуль виявлено при спробі поділити на {i}|{i}-ий елемент")
            for j in range(i-1, -1, -1):
                m = x[j][i] / x[i][i]
                for l in range(len(x[i])):
                    x[j][l] -= x[i][l] * m
        for i in range(len(x)):
            if x[i][i] == 0:
                raise ValueError(f"Ділення на нуль виявлено при спробі поділити на {i}|{i}-ий елемент")
            diag = x[i][i]
            for l in range(len(x[i])):
                x[i][l] /= diag
        last_column = [row[-1] for row in x]
        return last_column

    @staticmethod
    def validate_extrapolation(season, moving_average, method, path, prd_name):
        col_name = 'usdprice'
        if moving_average:
            col_name = 'moving_average'
        to_extr = Extrapolation.__loading(prd_name, path)

        if moving_average:
            col_name = 'moving_average'
            to_extr = to_extr.iloc[1:].reset_index(drop=True)

        # Train / Control split
        train_to_extr = to_extr[:int(len(to_extr) * 0.8)].copy()
        control_to_extr = to_extr[int(len(to_extr) * 0.8):].copy()

        # Початкова дата для обчислення коефіцієнтів
        start_date = train_to_extr['date'].min()
        start_year = start_date.year
        start_month = start_date.month

        # Обчислення datecoef
        train_to_extr['datecoef'] = ((train_to_extr['date'].dt.year - start_year) * 12) + \
                                    (train_to_extr['date'].dt.month - start_month)
        control_to_extr['datecoef'] = ((control_to_extr['date'].dt.year - start_year) * 12) + \
                                      (control_to_extr['date'].dt.month - start_month)

        # Побудова моделі
        x = train_to_extr['datecoef'].to_numpy()
        y = train_to_extr[col_name].to_numpy()
        y = [[value] for value in y]

        X = []
        if method == "Polinomial":
            for i in x:
                X.append([i ** 2, i, 1])
        else:
            for i in x:
                X.append([i, 1])

        X = []
        if method == "Polinomial":
            for i in x:
                X.append([i**2,i, 1])
        else:
            for i in x:
                X.append([i, 1])
        X_T = []
        for i in range(len(X[0])):
            row = []
            for j in range(len(X)):
                row.append(X[j][i])
            X_T.append(row)

        XXT = []
        for i in range(len(X_T)):
            row = []
            for j in range(len(X[0])):
                s = 0
                for k in range(len(X)):
                    s += X_T[i][k] * X[k][j]
                row.append(s)
            XXT.append(row)

        XTy = []
        for i in range(len(X_T)):
            s = 0
            for j in range(len(y)):
                s += X_T[i][j] * y[j][0]
            XTy.append(s)
        alignment = []

        for i in range(len(XXT)):
            row = XXT[i] + [XTy[i]]
            alignment.append(row)
        coefs = Extrapolation.gausian(alignment)

        # Прогноз на контрольні дати
        ctrl_x = control_to_extr['datecoef'].to_numpy()
        amp = 0.05
        freq = 2 * np.pi / 12
        phase = 0
        if method == "Polinomial":
            if season:
                s = amp * np.sin(freq * ctrl_x + phase)
                predicted_ctrl = coefs[0] * ctrl_x ** 2 + coefs[1] * ctrl_x + coefs[2] + s
            else:
                predicted_ctrl = coefs[0] * ctrl_x ** 2 + coefs[1] * ctrl_x + coefs[2]
        else:
            if season:
                s = amp * np.sin(freq * ctrl_x + phase)
                predicted_ctrl = coefs[0] * ctrl_x + coefs[1] + s
            else:
                predicted_ctrl = coefs[0] * ctrl_x + coefs[1]

        # Таблиця результатів
        real_vs_pred = control_to_extr[['date', col_name]].copy()
        real_vs_pred['predicted'] = predicted_ctrl

        real_vs_pred[col_name] = real_vs_pred[col_name].round(5)
        real_vs_pred['predicted'] = real_vs_pred['predicted'].round(5)
        predicted = real_vs_pred['predicted'].to_list()
        real = real_vs_pred[col_name].to_list()
        mse = 0
        for i in range(len(predicted)):
            mse += (real[i] - predicted[i])**2

        mse /= len(predicted)

        return real_vs_pred, mse

    @staticmethod
    def run_extrapolation(moving_average, season, method, end, path, prd_name, qt_parent):
        col_name = 'usdprice'
        to_extr = Extrapolation.__loading(prd_name, path)

        if moving_average:
            col_name = 'moving_average'
            to_extr = to_extr.iloc[1:].reset_index(drop=True)

        last_row = to_extr[['date', col_name]].tail(1)
        start_date = to_extr['date'].min()
        start_year = start_date.year
        start_month = start_date.month

        prediction_start_date = to_extr['date'].max().strftime('%Y-%m')
        prediction_end_date = end[:len(end)-3]

        to_extr['datecoef'] = ((to_extr['date'].dt.year-start_year)*12)+(to_extr['date'].dt.month-start_month)
        x = to_extr['datecoef'].to_numpy()
        y = to_extr[col_name].to_numpy()
        y = [[value] for value in y]

        X = []
        if method == "Polinomial":
            for i in x:
                X.append([i**2,i, 1])
        else:
            for i in x:
                X.append([i, 1])
        X_T = []
        for i in range(len(X[0])):
            row = []
            for j in range(len(X)):
                row.append(X[j][i])
            X_T.append(row)

        XXT = []
        for i in range(len(X_T)):
            row = []
            for j in range(len(X[0])):
                s = 0
                for k in range(len(X)):
                    s += X_T[i][k] * X[k][j]
                row.append(s)
            XXT.append(row)

        XTy = []
        for i in range(len(X_T)):
            s = 0
            for j in range(len(y)):
                s += X_T[i][j] * y[j][0]
            XTy.append(s)
        alignment = []

        for i in range(len(XXT)):
            row = XXT[i] + [XTy[i]]
            alignment.append(row)
        coefs = Extrapolation.gausian(alignment)
        if method == "Polinomial":
            extr, inflation_over_all_period = Extrapolation.predict_with_date(col_name, season, last_row, to_extr['date'].max(), end, start_year, start_month, coefs[0], coefs[1], coefs[2])
        else:
            extr, inflation_over_all_period = Extrapolation.predict_with_date(col_name, season, last_row, to_extr['date'].max(), end, start_year, start_month, None, coefs[0], coefs[1])

        fig, ax = plt.subplots(figsize=(12, 3))
        combined_dates = pd.concat([to_extr['date'], extr['date']])
        combined_prices = pd.concat([to_extr[col_name], extr['predicted_usdprice']])

        ax.plot(combined_dates, combined_prices, marker='o', linestyle='-', color='blue')
        ax.plot(extr['date'], extr['predicted_usdprice'], marker='o', linestyle='-', color='red')

        trend_x = to_extr['datecoef'].to_numpy()
        trend_dates = to_extr['date']
        amp = 0.05
        freq = 2 * np.pi / 12
        phase = 0
        # Обчислюємо тренд
        if method == "Polinomial":
            if season:
                s = amp * np.sin(freq * trend_x+ phase)
                trend_y = coefs[0] * trend_x ** 2 + coefs[1] * trend_x + coefs[2] + s
            else:
                trend_y = coefs[0] * trend_x ** 2 + coefs[1] * trend_x + coefs[2]
        else:
            if season:
                s = amp * np.sin(freq * trend_x + phase)
                trend_y = coefs[0] * trend_x + coefs[1] + s
            else:
                trend_y = coefs[0] * trend_x + coefs[1]

        ax.plot(trend_dates, trend_y, linestyle='-', color='green')

        ax.set_title('Predicted USD Price Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('USD Price')

        inflation_text = (
            f"Інфляція за період {prediction_start_date} → {prediction_end_date}: "
            f"{'+' if inflation_over_all_period > 0 else ''}{inflation_over_all_period:.4f}%"
        )

        ax.text(
            0.01, 0.95,
            inflation_text,
            transform=ax.transAxes,
            ha='left',
            va='top',
            color='black',
            fontsize=12
        )
        ax.grid(True)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setParent(qt_parent)
        return canvas, fig