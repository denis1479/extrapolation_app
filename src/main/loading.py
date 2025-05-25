import pandas as pd
import os


class LoadingAndAggregation:
    country_codes = ['ukr', 'afg', 'ago', 'arg', 'arm', 'aze', 'bdi', 'ben', 'bfa', 'bgd',
                     'bol', 'btn', 'caf', 'chn', 'civ', 'cmr', 'cod', 'cog', 'col', 'cpv',
                     'dji', 'dom', 'dza', 'ecu', 'eri', 'eth', 'fji', 'gab', 'geo', 'gha',
                     'gin', 'gmb', 'gnb', 'gtm', 'hnd', 'hti', 'idn', 'ind', 'irn', 'irq',
                     'jpn', 'kaz', 'ken', 'kgz', 'khm', 'lao', 'lbn', 'lbr', 'lby', 'lka',
                     'mda', 'mdg', 'mex', 'mli', 'mmr', 'mng', 'moz', 'mrt', 'mwi', 'nam',
                     'nga', 'nic', 'npl', 'pak', 'pan', 'per', 'phl', 'pry', 'pse', 'rwa',
                     'sen', 'sle', 'slv', 'som', 'ssd', 'swz', 'syr', 'tcd', 'tgo', 'tha',
                     'tls', 'tur', 'tza', 'uga', 'vnm', 'yem', 'zaf', 'zmb', 'zwe', 'cri',
                     'jor', 'lso', 'ner', 'tjk']

    EXCLUDED_COUNTRIES = {
        'afg', 'arg', 'eth', 'gha', 'hti', 'irn', 'lbn', 'lby',
        'nga', 'pak', 'sdn', 'ssd', 'syr', 'ven', 'zmb', 'zwe',
        'fji', 'lso', 'mwi', 'gnb', 'bfa', 'gmb', 'eri', 'syr'
        'kgz'
    }
    product_names = ['Bread', 'Meat', 'Sugar', 'Fish', 'Eggs', 'Rice', 'Oil', 'Milk', 'Potatoes', 'Buckwheat',
                     'Carrots', 'Beetroots', 'Onions', 'Wheat flour']

    def __new__(cls, *args, **kwargs):
        raise TypeError("це статичний клас")

    @staticmethod
    def __standardize_commodity_name(commodity):
        for product in LoadingAndAggregation.product_names:
            if commodity.startswith(product):
                return product
        return commodity

    @staticmethod
    def processing(country_code):
        df = pd.read_csv(f"{os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")}/wfp_food_prices_{country_code}.csv", low_memory=False, skiprows=[1])
        products_units = {
            "Bread": "KG",
            "Meat": "KG",
            "Sugar": "KG",
            "Fish": "KG",
            "Eggs": "10 pcs",
            "Rice": "KG",
            "Oil": "L",
            "Milk": "L",
            "Potatoes": "KG",
            "Buckwheat": "KG",
            "Carrots": "KG",
            "Beetroots": "KG",
            "Onions": "KG",
            "Wheat flour": "KG"
        }
        df = df[df['date'].str.match(r'\d{4}-\d{2}-\d{2}', na=False)]
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.astype({'category': 'string', 'usdprice': 'float'})
        grouped_df = df[df.apply(lambda row: any(row["commodity"].startswith(prod) and row["unit"] == unit for prod, unit in products_units.items()), axis=1)]
        grouped_df.loc[:, "commodity"] = grouped_df["commodity"].apply(LoadingAndAggregation.__standardize_commodity_name)
        df_filtered = grouped_df[((grouped_df['date'] >= '2020-01-01')) & (grouped_df['date'] < '2025-01-01')]
        # Сортуємо по даті
        df_sorted = df_filtered.sort_values(by='date')
        return df_sorted

    @staticmethod
    def noaggregation():
        country_codes = [code for code in LoadingAndAggregation.country_codes if code not in LoadingAndAggregation.EXCLUDED_COUNTRIES]
        dfs = []
        for i in country_codes:
            dfs.append(LoadingAndAggregation.processing(i))
        if not dfs:
            return pd.DataFrame()
        df_concat = pd.concat(dfs, ignore_index=True)
        df_concat_sorted = df_concat.sort_values(by='date')
        # Видаляємо None та дублікати
        df_concat_sorted = df_concat_sorted.dropna()
        df_concat_sorted = df_concat_sorted.drop_duplicates()

        # Групуємо по місяцям та продуктам для видалення аномальних значень
        grouped = df_concat_sorted.groupby(['date', 'commodity'])
        dfs_by_date_commodity = {date: data for date, data in grouped}
        filted_dfs_by_date_commodity = {}
        for key, value in dfs_by_date_commodity.items():
            prices = value['usdprice'].sort_values().to_list()
            n = len(prices)

            # Розділення на нижню і верхню половини
            if n % 2 == 0:
                lower_half = prices[:n // 2]
                upper_half = prices[n // 2:]
            else:
                lower_half = prices[:n // 2]
                upper_half = prices[n // 2 + 1:]

            # Обчислення Q1
            l1 = len(lower_half)
            if l1 % 2 == 0:
                Q1 = (lower_half[l1 // 2 - 1] + lower_half[l1 // 2]) / 2
            else:
                Q1 = lower_half[l1 // 2]

            # Обчислення Q3
            l3 = len(upper_half)
            if l3 % 2 == 0:
                Q3 = (upper_half[l3 // 2 - 1] + upper_half[l3 // 2]) / 2
            else:
                Q3 = upper_half[l3 // 2]

            # Інтерквартильний розмах (IQR)
            IQR = Q3 - Q1
            left_limit = Q1 - 1.5 * IQR
            right_limit = Q3 + 1.5 * IQR
            df_filtered = value[(value['usdprice'] > left_limit) & (value['usdprice'] < right_limit)]
            filted_dfs_by_date_commodity[key] = df_filtered

        if not filted_dfs_by_date_commodity:
            return pd.DataFrame()
        df_concat_cleared = pd.concat(filted_dfs_by_date_commodity, ignore_index=True)

        df_concat_sorted_cleared = df_concat_cleared.sort_values(by='date')
        dfs_by_commodity = {}
        for product_name in LoadingAndAggregation.product_names:
            dfs_by_commodity[product_name] = df_concat_sorted_cleared[['date', 'usdprice']][df_concat_sorted_cleared['commodity'].str.startswith(product_name, na=False)]
        return dfs_by_commodity

    @staticmethod
    def aggregation():

        country_codes = [code for code in LoadingAndAggregation.country_codes if code not in LoadingAndAggregation.EXCLUDED_COUNTRIES]
        dfs = []
        for i in country_codes:
            to_add = LoadingAndAggregation.processing(i)
            to_add['country'] = i
            dfs.append(to_add)

        if not dfs:
            return pd.DataFrame()

        df_concat = pd.concat(dfs, ignore_index=True)
        df_concat_sorted = df_concat.sort_values(by='date')

        # Видаляємо None та дублікати
        df_concat_sorted = df_concat_sorted.dropna()
        df_concat_sorted = df_concat_sorted.drop_duplicates()

        # Групуємо по місяцям та продуктам для видалення аномальних значень
        grouped = df_concat_sorted.groupby(['date', 'commodity'])
        dfs_by_date = {date: data for date, data in grouped}
        # Для кожного місяця знаходимо та видаляємо аномалії за допомогою IQR та Z-оцінки.
        filted_dfs_by_date = {}
        for key, value in dfs_by_date.items():
            prices = value['usdprice'].sort_values().to_list()
            n = len(prices)

            # Розділення на нижню і верхню половини
            if n % 2 == 0:
                lower_half = prices[:n // 2]
                upper_half = prices[n // 2:]
            else:
                lower_half = prices[:n // 2]
                upper_half = prices[n // 2 + 1:]

            # Обчислення Q1
            l1 = len(lower_half)
            if l1 % 2 == 0:
                Q1 = (lower_half[l1 // 2 - 1] + lower_half[l1 // 2]) / 2
            else:
                Q1 = lower_half[l1 // 2]

            # Обчислення Q3
            l3 = len(upper_half)
            if l3 % 2 == 0:
                Q3 = (upper_half[l3 // 2 - 1] + upper_half[l3 // 2]) / 2
            else:
                Q3 = upper_half[l3 // 2]

            # Інтерквартильний розмах (IQR)
            IQR = Q3 - Q1
            left_limit = Q1 - 1.5 * IQR
            right_limit = Q3 + 1.5 * IQR
            df_filtered = value[(value['usdprice'] > left_limit) & (value['usdprice'] < right_limit)]
            filted_dfs_by_date[key] = df_filtered
        if not filted_dfs_by_date:
            return pd.DataFrame()
        df_concat_cleared = pd.concat(filted_dfs_by_date, ignore_index=True)
        df_grouped = df_concat_cleared[['country', 'date', 'commodity', 'usdprice']].groupby(['country', 'date', 'commodity'])['usdprice'].mean().reset_index()
        # Групуємо по місяцям та продуктам для видалення аномальних значень
        grouped = df_grouped.groupby(['date', 'commodity'])
        dfs_by_date = {date: data for date, data in grouped}
        # Для кожного місяця знаходимо та видаляємо аномалії за допомогою IQR та Z-оцінки.
        filted_dfs_by_date = {}
        for key, value in dfs_by_date.items():
            Q1 = value['usdprice'].quantile(0.25)
            Q3 = value['usdprice'].quantile(0.75)

            # Інтерквартильний розмах (IQR)
            IQR = Q3 - Q1
            left_limit = Q1 - 1.5 * IQR
            right_limit = Q3 + 1.5 * IQR
            df_filtered = value[(value['usdprice'] > left_limit) & (value['usdprice'] < right_limit)]
            filted_dfs_by_date[key] = df_filtered
        if not filted_dfs_by_date:
            return pd.DataFrame()
        df_concat_cleared = pd.concat(filted_dfs_by_date, ignore_index=True)
        df_concat_sorted_cleared = df_concat_cleared.sort_values(by='date')
        dfs_by_commodity = {}
        for product_name in LoadingAndAggregation.product_names:
            dfs_by_commodity[product_name] = df_concat_sorted_cleared[['date', 'usdprice']][df_concat_sorted_cleared['commodity'].str.startswith(product_name, na=False)]

        dfs_by_commodity_result = {}
        for commodity, data in dfs_by_commodity.items():
            monthly_avg = data[['date', 'usdprice']].groupby(['date'])['usdprice'].median().reset_index()
            prices = monthly_avg['usdprice'].tolist()
            inflation_rates = [None]  # перше значення не має попереднього

            for i in range(1, len(prices)):
                prev = prices[i - 1]
                curr = prices[i]
                rate = (curr / prev - 1) if prev != 0 else None
                inflation_rates.append(rate)

            monthly_avg['inflation_rate'] = inflation_rates
            # Ручна реалізація плинного середнього з вікном 3
            moving_averages = []
            window_size = 3
            prices = monthly_avg['usdprice'].tolist()

            for i in range(len(prices)):
                window = prices[max(0, i - window_size + 1):i + 1]  # останні window_size значень
                avg = sum(window) / len(window)
                moving_averages.append(avg)
            if len(moving_averages) == len(monthly_avg) and len(moving_averages) > 0:
                monthly_avg['moving_average'] = moving_averages

                dfs_by_commodity_result[commodity] = monthly_avg

        return dfs_by_commodity_result

    @staticmethod
    def loading(df, path):
        for product_name in LoadingAndAggregation.product_names:
            df[product_name].to_csv(f'{path}/{product_name.replace(' ', '_')}.csv', index=False,
                                                         encoding='utf-8')

    @staticmethod
    def run_pipeline(path_processed, path_raw, test=False):
        if test:
            no_agg = LoadingAndAggregation.noaggregation()
            agg = LoadingAndAggregation.aggregation()
            return no_agg, agg
        LoadingAndAggregation.loading(LoadingAndAggregation.noaggregation(), path_raw)
        LoadingAndAggregation.loading(LoadingAndAggregation.aggregation(), path_processed)