import pytest
from unittest.mock import patch
import pandas as pd
from math import isclose
from src.main.loading import LoadingAndAggregation


@pytest.fixture
def sample_data():
    df_mock_test_country_1 = pd.DataFrame({
        'date': ['2020-02-15', '2020-03-15', '2020-02-15', '2020-02-15'],
        'admin1': ['C1_TestAdmin1_1', 'C1_TestAdmin1_2', 'C1_TestAdmin1_2', 'C1_TestAdmin1_2'],
        'admin2': ['C1_TestAdmin2_1', 'C1_TestAdmin2_2', 'C1_TestAdmin2_2', 'C1_TestAdmin2_2'],
        'market': ['C1_TestMarket_1', 'C1_TestMarket_2', 'C1_TestMarket_3', 'C1_TestMarket_3'],
        'latitude': [55.0, 55.0, 33.0, 22.0],
        'longitude': [-40.0, 70.0, 44.2, 55.4],
        'category': ['C1_TestCategory_1', 'C1_TestCategory_2', 'C1_TestCategory_2', 'C1_TestCategory_2'],
        'commodity': ['Bread 1', 'Meat 2', 'Meat 2', 'Meat 2'],
        'unit': ['KG', 'KG', 'KG', 'KG'],
        'priceflag': ['C1_TestPriceFlag_1', 'C1_TestPriceFlag_1', 'C1_TestPriceFlag_1', 'C1_TestPriceFlag_1'],
        'pricetype': ['C1_TestPriceType_1', 'C1_TestPriceType_1', 'C1_TestPriceType_1', 'C1_TestPriceType_1'],
        'currency': ['TESTCURC1', 'TESTCURC1', 'TESTCURC1', 'TESTCURC1'],
        'price': [200.0, 300.0, 250.0, 150.0],
        'usdprice': [5.0, 7.0, 6.0, 5.0]
    })
    df_mock_test_country_2 = pd.DataFrame({
        'date': ['2020-02-15', '2020-03-15', '2020-02-15'],
        'admin1': ['C2_TestAdmin1_1', 'C2_TestAdmin1_2', 'C2_TestAdmin1_2'],
        'admin2': ['C2_TestAdmin2_1', 'C2_TestAdmin2_2', 'C2_TestAdmin2_2'],
        'market': ['C2_TestMarket_1', 'C2_TestMarket_2', 'C2_TestMarket_2'],
        'latitude': [50.0, 60.0, 40.0],
        'longitude': [-30.0, 50.0, 30.0],
        'category': ['C2_TestCategory_1', 'C2_TestCategory_2', 'C2_TestCategory_1'],
        'commodity': ['Bread 1', 'Meat 2', 'Meat 2'],
        'unit': ['KG', 'KG', 'KG'],
        'priceflag': ['C2_TestPriceFlag_1', 'C2_TestPriceFlag_1', 'C2_TestPriceFlag_1'],
        'pricetype': ['C2_TestPriceType_1', 'C2_TestPriceType_1', 'C2_TestPriceType_1'],
        'currency': ['TESTCURC2', 'TESTCURC2', 'TESTCURC2'],
        'price': [100.0, 200.0, 600.0],
        'usdprice': [10.0, 20.0, 19.0]
    })
    return df_mock_test_country_1, df_mock_test_country_2


@pytest.fixture
def patch_country_codes():
    with patch.object(LoadingAndAggregation, 'country_codes', new=['TestCountry_1', 'TestCountry_2']):
        yield


@pytest.fixture
def patch_loading():
    with patch.object(LoadingAndAggregation, 'loading', side_effect=lambda df, path=None: df):
        yield


@pytest.fixture
def patch_read_csv(sample_data):
    df_mock_test_country_1, df_mock_test_country_2 = sample_data
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.side_effect = [
            df_mock_test_country_1,
            df_mock_test_country_2,
            df_mock_test_country_1,
            df_mock_test_country_2
        ]
        yield mock_read_csv


def test_run_pipeline_(patch_country_codes, patch_loading, patch_read_csv):
    result_no_agg, result_agg = LoadingAndAggregation.run_pipeline('test_path', 'test_path', True)

    # Перевірки агрегованих даних
    assert not result_agg['Bread'].empty, "Агрегований датасет Bread не пустий"
    assert not result_agg['Meat'].empty, "Агрегований датасет Meat не пустий"

    assert len(result_no_agg['Meat']['usdprice']) == 5, "Розмір неагрегованих даних Meat - 5"
    assert len(result_agg['Meat']['usdprice']) == 2, "Розмір агрегованих даних Meat - 2"

    expected_prices_meat_noagr = [6.0, 5.0, 19.0, 7.0, 20.0]
    expected_prices_meat_agr = [12.25, 13.50]
    assert result_no_agg['Meat']['usdprice'].to_list() == expected_prices_meat_noagr
    assert result_agg['Meat']['usdprice'].to_list() == expected_prices_meat_agr

    expected_inflation_rate_agr = (13.50 - 12.25) / 12.25
    assert isclose(result_agg['Meat']['inflation_rate'].to_list()[1], expected_inflation_rate_agr, rel_tol=1e-9)

    # Перевірка формату дати
    for product, df in result_agg.items():
        assert pd.api.types.is_datetime64_any_dtype(df['date']), f"Колонка date у {product} має бути datetime"

    # Перевірка назв продуктів (без цифр)
    for d in [result_no_agg, result_agg]:
        for product in d.keys():
            assert isinstance(product, str) and not any(char.isdigit() for char in product), \
                f"Назва продукту '{product}' має бути рядком без цифр"


@pytest.mark.parametrize("empty_dfs", [True])
def test_empty_data(empty_dfs, patch_country_codes, patch_loading):
    # Порожні DataFrame
    empty_df = pd.DataFrame(columns=[
        'date', 'admin1', 'admin2', 'market', 'latitude', 'longitude', 'category',
        'commodity', 'unit', 'priceflag', 'pricetype', 'currency', 'price', 'usdprice'
    ])
    with patch("pandas.read_csv") as mock_csv:
        mock_csv.side_effect = [empty_df, empty_df, empty_df, empty_df]
        res_no_agg, res_agg = LoadingAndAggregation.run_pipeline('test_path', 'test_path', True)
        for product in res_no_agg:
            assert res_no_agg[product].empty
        for product in res_agg:
            assert res_agg[product].empty

@pytest.fixture
def patch_country_codes_single():
    with patch.object(LoadingAndAggregation, 'country_codes', new=['TestCountry_1']):
        yield

@pytest.mark.parametrize("same_price", [True])
def test_inflation_zero(same_price, patch_country_codes, patch_loading, sample_data):
    df_mock_test_country_1, df_mock_test_country_2 = sample_data
    df_mock_test_country_1['usdprice'] = 10.0
    df_mock_test_country_2['usdprice'] = 10.0
    with patch("pandas.read_csv") as mock_csv:
        mock_csv.side_effect = [
            df_mock_test_country_1,
            df_mock_test_country_2,
            df_mock_test_country_1,
            df_mock_test_country_2
        ]
        res_no_agg, res_agg = LoadingAndAggregation.run_pipeline('test_path', 'test_path', True)
        for product in res_agg:
            inflation_values = res_agg[product]['inflation_rate'].dropna().to_list()
            for val in inflation_values:
                assert isclose(val, 0.0, abs_tol=1e-9)


