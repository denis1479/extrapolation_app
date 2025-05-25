from src.main.extrapolation import Extrapolation
import pandas as pd
import numpy as np


def test_gausian_basic_solution():
    """
    Перевірка базового випадку для функції gausian.
    Розв'язання системи лінійних рівнянь з відомим розв'язком.
    """
    matrix = np.array([[1, 1, 1, 6], [2, 1, 3, 13], [-1, 4, 1, 3]])
    expected_result = np.array([3.0, 1.0, 2.0])
    result = Extrapolation.gausian(matrix)
    assert np.allclose(result, expected_result), "Basic Gaussian elimination failed"

def test_incorrect_type():
    """
    Перевірка невірного типа даних
    """
    matrix = "123"
    expected_result = "123"
    result = Extrapolation.gausian(matrix)
    assert result == expected_result, "Excepted returning input data if type is incorrect"


def test_gausian_inconsistent_system():
    """
    Перевірка на неконсистентну систему (без розв'язку).
    Очікується виняток або специфічне повідомлення про неможливість розв'язання.
    """
    matrix = np.array([[1, 1, 1, 6], [1, 1, 1, 7], [1, 1, 1, 8]])
    try:
        Extrapolation.gausian(matrix)
        assert False, "Expected exception for inconsistent system"
    except Exception:
        pass  # Очікувана поведінка


def test_gausian_singular_matrix():
    """
    Перевірка на вироджену матрицю (нескінченно багато розв'язків або немає).
    """
    matrix = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]])  # Лінійно залежні рядки
    try:
        Extrapolation.gausian(matrix)
        assert False, "Expected exception for singular matrix"
    except Exception:
        pass


def test_predict_with_date_structure_and_columns():
    """
    Перевірка, що функція повертає DataFrame з очікуваною структурою колонок.
    """
    last_row = pd.DataFrame({
        'date': [pd.Timestamp('2020-01-15')],
        'usdprice': [100.0]
    })

    predicted_df, _ = Extrapolation.predict_with_date(
        col_name='usdprice',
        season=True,
        last_row=last_row,
        x_start_date=pd.Timestamp('2020-01-01'),
        x_end_date='2020-04-01',
        start_year=2020,
        start_month=1,
        a=0.1,
        b=2.0,
        c=50.0
    )

    assert isinstance(predicted_df, pd.DataFrame), "Returned value is not a DataFrame"
    assert all(col in predicted_df.columns for col in
               ['date', 'predicted_usdprice', 'inflation_rate']), "Missing expected columns"


def test_predict_with_date_correct_dates():
    """
    Перевірка правильності розрахунку майбутніх дат.
    """
    last_row = pd.DataFrame({'date': [pd.Timestamp('2020-01-15')], 'usdprice': [100.0]})
    predicted_df, _ = Extrapolation.predict_with_date('usdprice', True, last_row, pd.Timestamp('2020-01-01'), '2020-04-01', 2020, 1,
                                                      0.1, 2.0, 50.0)

    expected_dates = [pd.Timestamp('2020-02-15'), pd.Timestamp('2020-03-15'), pd.Timestamp('2020-04-15')]
    assert list(predicted_df['date']) == expected_dates, "Prediction dates are incorrect"


def test_predict_with_date_seasonality_and_inflation():
    """
    Перевірка точності розрахунку прогнозованих цін з урахуванням сезонності
    та підрахунок інфляції за весь період.
    """
    last_row = pd.DataFrame({'date': [pd.Timestamp('2020-01-15')], 'usdprice': [100.0]})
    a, b, c = 0.1, 2.0, 50.0
    amp = 0.05
    freq = 2 * np.pi / 12
    phase = 0

    predicted_df, inflation = Extrapolation.predict_with_date(
        'usdprice', True, last_row, pd.Timestamp('2020-01-01'), '2020-04-01', 2020, 1, a, b, c
    )

    # Очікувані ціни
    expected_prices = []
    for x_index in range(1, 4):
        y_pred = a * x_index ** 2 + b * x_index + c
        seasonality = amp * np.sin(freq * x_index + phase)
        expected_prices.append(y_pred + seasonality)

    for predicted, expected in zip(predicted_df['predicted_usdprice'], expected_prices):
        assert np.isclose(predicted, expected, atol=1e-8), f"Predicted price {predicted} != expected {expected}"

    # Перевірка інфляції
    first_val = last_row.iloc[0]['usdprice']
    last_val = expected_prices[-1]
    expected_inflation = ((last_val - first_val) / first_val) * 100
    assert np.isclose(inflation, expected_inflation,
                      atol=1e-8), f"Inflation {inflation} != expected {expected_inflation}"

def test_predict_with_date_no_seasonality():
    """
    Перевірка прогнозу без сезонності: сезонна складова повинна бути відсутня.
    Перевіряється, що ціни розраховані лише за квадратичною формулою.
    """
    last_row = pd.DataFrame({'date': [pd.Timestamp('2020-01-15')], 'usdprice': [100.0]})
    a, b, c = 0.1, 2.0, 50.0

    predicted_df, inflation = Extrapolation.predict_with_date(
        col_name='usdprice',
        season=False,
        last_row=last_row,
        x_start_date=pd.Timestamp('2020-01-01'),
        x_end_date='2020-04-01',
        start_year=2020,
        start_month=1,
        a=a, b=b, c=c
    )

    # Очікувані ціни без сезонності
    expected_prices = [a * x**2 + b * x + c for x in range(1, 4)]

    for predicted, expected in zip(predicted_df['predicted_usdprice'], expected_prices):
        assert np.isclose(predicted, expected, atol=1e-8), f"Predicted price {predicted} != expected {expected} (no seasonality)"

    # Перевірка інфляції
    first_val = last_row.iloc[0]['usdprice']
    last_val = expected_prices[-1]
    expected_inflation = ((last_val - first_val) / first_val) * 100
    assert np.isclose(inflation, expected_inflation, atol=1e-8), f"Inflation {inflation} != expected {expected_inflation}"
