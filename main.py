import os
import pandas as pd
from jupyterthemes import jtplot
jtplot.reset()
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.preprocessing import LabelEncoder
import phik

st.set_page_config(layout="wide")


# распаковываем датасеты 
folder_path = r'/Users/darkstrouk/Documents/МОВС Магистратура/Прикладной Python/hw_1/stepik_linear_models/datasets'

cnt = 1
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        file_name_without_extension = os.path.splitext(file_name)[0]
        globals()[f'df_{file_name_without_extension}'] = pd.read_csv(file_path)
        cnt += 1


# чистим данные
df_list = [df_D_clients, df_D_job, df_D_last_credit, df_D_loan, df_D_salary, df_D_target, df_D_work, df_D_close_loan, df_D_pens]

for df in df_list:
    #удаляем пропуски
    for index, row in df.iterrows():
        df.loc[index] = row.dropna()
    #удаляем дубли чисто по айди
    for column in df.columns:
        if column.startswith('ID'):  
            df.drop_duplicates(subset=column, inplace=True)
    #удаляем аномалии
    numeric_columns = df.select_dtypes(include=['int', 'float'])
    # numeric_columns = numeric_columns.loc[:, ~numeric_columns.columns.str.startswith(('ID', 'FLAG'))]
    for column in numeric_columns:
        mean = df[column].mean()
        std = df[column].std()
        lower_limit = mean - 3 * std
        upper_limit = mean + 3 * std
        df[column] = np.where((df[column] < lower_limit) | (df[column] > upper_limit), np.nan, df[column])


# склеиваем данные
df = df_D_clients.merge(df_D_job, left_on='ID', right_on='ID_CLIENT', how='left')
df = df.merge(df_D_last_credit, on='ID_CLIENT', how='left')
df = df.merge(df_D_loan, on='ID_CLIENT', how='left')
df = df.merge(df_D_salary, on='ID_CLIENT', how='left')
df = df.merge(df_D_target, on='ID_CLIENT', how='left')
df = df.merge(df_D_work, left_on='SOCSTATUS_WORK_FL', right_on='ID', how='left')
df = df.merge(df_D_close_loan, left_on='ID_LOAN', right_on='ID_LOAN', how='left')
df = df.merge(df_D_pens, left_on='SOCSTATUS_PENS_FL', right_on='ID', how='left')


# заполняем пропуски средним значением
for column in numeric_columns:
    if numeric_columns[column].isnull().sum() > 0:  
        mean_value = numeric_columns[column].mean()  
        numeric_columns[column].fillna(mean_value, inplace=True) 


# удаляем ненужные колонки
df.drop(['ID_CLIENT', 'ID_LOAN', 'ID_y', 'FLAG_x', 'ID', 'FLAG_y', 'ID_x'], axis=1, inplace=True)





# САМО ПРИЛОЖЕНИЕ

st.title('Разведочный анализ данных')
st.subheader('Перед визуализацей данные были предобработанны - отчищены от пропусков, аномалий и дублей')


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Матрица корреляций", 'Пустые значения', "Распределение признаков", 'Зависимость целевой переменной от признаков', 'Статистика числовых данных'])


# матрица корреляций
with tab1:
    st.subheader('Матрица Корреляций')

    df_matrix = df.copy()
    label_encoder = LabelEncoder()

    for column in df_matrix.columns:
        if df_matrix[column].dtype == 'object':
            df_matrix[column] = label_encoder.fit_transform(df_matrix[column])

    for column in df_matrix.columns:
        if df_matrix[column].isnull().sum() > 0:  
            mean_value = df_matrix[column].mean()  
            df_matrix[column].fillna(mean_value, inplace=True) 

    corr_matrix = df_matrix.phik_matrix()
    corr_matrix_long = corr_matrix.reset_index().melt('index')
    corr_matrix_long.columns = ['Variable 1', 'Variable 2', 'correlation']


    # Отображение тепловой карты
    corr_chart = alt.Chart(corr_matrix_long).mark_rect().encode(
        alt.X('Variable 1:N', title=None),
        alt.Y('Variable 2:N', title=None),
        color=alt.Color('correlation:Q', scale=alt.Scale(scheme='brownbluegreen')),
        tooltip=[
            alt.Tooltip('Variable 1:N', title='Variable 1'),
            alt.Tooltip('Variable 2:N', title='Variable 2'),
            alt.Tooltip('correlation:Q', title='Correlation')
        ]
    ).properties(
        width=1000,
        height=1000
    )

    # Добавление текста с значениями корреляции
    text = alt.Chart(corr_matrix_long).mark_text(baseline='middle').encode(
        x='Variable 1:N',
        y='Variable 2:N',
        text=alt.Text('correlation:Q', format='.1f'),
        color=alt.condition(
            "datum.correlation > 0.5",
            alt.value('white'),
            alt.value('black')
        )
    )

    # Комбинация тепловой карты и текстовых меток
    final_chart = alt.layer(corr_chart, text).resolve_scale(color='independent').configure_axis(
        domain=False,
        tickSize=0
    )

    # Отображение графика в Streamlit
    st.altair_chart(final_chart, use_container_width=True)

with tab2:
    # "Длинный" формат DataFrame
    df_melted = pd.melt(df.reset_index(), id_vars=['index'], var_name='column', value_name='value')

    # Визуализация heatmap с указанными параметрами
    heatmap = alt.Chart(df_melted).mark_rect().encode(
        alt.X('column:N', axis=alt.Axis(title='')),
        alt.Y('index:N', axis=alt.Axis(title='')),
        color=alt.condition(
            'isValid(datum.value)',
            alt.value('#7f4d10'),  # для непропущенных значений
            alt.value('#1a7971')  # для пропущенных значений
        ),
        tooltip=['column:N', 'index:N', 'value:Q']
    ).properties(
        title={
            "text": ["Пропуски в данных"],
            "subtitle": ["Зеленый - данные присутствуют, Коричневый - пропуски в данных"],
            "color": "white",
            "subtitleColor": "white"
        }
    ).configure_title(
        fontSize=20,
        anchor='start',
        color='white'  # Белый цвет для заголовка и подзаголовка
    )

    # Выводим визуализацию в Streamlit
    st.altair_chart(heatmap, use_container_width=True)

with tab3:
    st.subheader('Распределение признаков')

    numeric_columns = df.select_dtypes(include=['int', 'float'])
    # numeric_columns = numeric_columns.loc[:, ~numeric_columns.columns.str.startswith(('ID', 'FLAG'))]

    for col in numeric_columns:
        # Определяем диапазон значений для осей
        scale = alt.Scale(domain=(numeric_columns[col].min(), numeric_columns[col].max()))
        
        # Создаем гистограмму
        hist = alt.Chart(numeric_columns).mark_bar(opacity=0.3, binSpacing=1).encode(
            alt.X(f"{col}:Q", bin=alt.Bin(maxbins=50), scale=scale, title=col),
            alt.Y('count()', stack=None, title='Количество'),
            tooltip=[col, alt.Tooltip('count()', title='Количество')]
        ).properties(
            width=600,
            height=200
        ).interactive()
        
        # Создаем оценку плотности ядра
        kde = alt.Chart(numeric_columns).transform_density(
            col,
            as_=[col, "density"],
        ).mark_area(opacity=0.3).encode(
            alt.X(f"{col}:Q", scale=scale, title=""),
            alt.Y('density:Q', title='Плотность'),
            tooltip=[alt.Tooltip(f"{col}:Q", title=col), alt.Tooltip('density:Q', title='Плотность')]
        ).properties(
            width=600,
            height=200
        )

        # Комбинируем гистограмму и оценку плотности ядра
        combined = alt.layer(hist, kde).resolve_scale(
            y='independent'
        )
        
        st.altair_chart(combined)

with tab4:    
    exclude_columns = ['REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE']

    df_feat = df.copy()

    def add_line_breaks(label):
        max_length = 10  
        return '\n'.join([label[i:i+max_length] for i in range(0, len(label), max_length)])


    def calculate_label_angle(df_feat, feature):
        if df_feat[feature].apply(len).max() <= 5:
            return 0
        else:
            return 270



    target = 'TARGET'
    features = [col for col in df_feat.columns if col != target]

    for feature in features:
        if feature in exclude_columns:
            continue  
        
        df_feat[feature] = df_feat[feature].astype(str)
        df_feat[feature] = df_feat[feature].apply(add_line_breaks)
        label_angles = calculate_label_angle(df_feat, feature)


        filtered_df = df_feat.dropna(subset=[feature, target])
        unique_values = len(filtered_df[feature].unique())

        if filtered_df[feature].dtype == 'object' or unique_values < 10:
            # Для категориальных признаков используем Bar Chart
            chart = alt.Chart(filtered_df).mark_bar().encode(
                x=alt.X(f'{feature}:O', axis=alt.Axis(labelAngle=label_angles)),  
                y='count():Q',
                color=f'{target}:N',
                tooltip=[feature, 'count()']
            ).interactive()

        elif unique_values > 30:
            # Для числовых признаков с множеством уникальных значений используем Histogram
            chart = alt.Chart(filtered_df).mark_bar().encode(
                x=alt.X(f'{feature}:O', axis=alt.Axis(labelAngle=label_angles), bin=True),  
                y='count():Q',
                color=f'{target}:N',
                tooltip=[feature, 'count()']
            ).interactive()

        else:
            # Для числовых признаков используем Box Plot
            chart = alt.Chart(filtered_df).mark_boxplot().encode(
                x=alt.X(f'{target}:O', axis=alt.Axis(labelAngle=label_angles)),
                y=f'{feature}:Q',
                tooltip=[feature, target]
            ).interactive()

        st.write(f"Зависимость {target} от {feature}")
        st.altair_chart(chart, use_container_width=True)

with tab5:

    numeric_columns = df.select_dtypes(include=['int', 'float'])

    # заполняем пропуски средним значением
    for column in numeric_columns:
        if numeric_columns[column].isnull().sum() > 0:  
            mean_value = numeric_columns[column].mean()  
            numeric_columns[column].fillna(mean_value, inplace=True) 



    st.title('Статистические характеристики DataFrame')
    column = st.selectbox('Выберите столбец для анализа', numeric_columns.columns.tolist())


    def compute_statistics(df):
        stats_df = df.describe().reset_index()
        return stats_df


    stats_df = compute_statistics(numeric_columns[[column]])

    st.write('Статистические характеристики для столбца:', column)
    st.dataframe(stats_df)



    boxplot_chart = alt.Chart(df).mark_boxplot(
        extent='min-max',  # Определяет, как далеко должны тянуться усы от коробки
    ).encode(
        y=alt.Y(column + ':Q', title=None),
        color=alt.value('#1a7971'),  # Устанавливает цвет заливки коробки
    ).properties(
        title=f'Boxplot для {column}',
        height=300,
        width=300
    )


    # Создание гистограммы
    hist_chart = alt.Chart(df).mark_bar(orient='horizontal').encode(
        x=alt.X('count()', title=None),
        y=alt.Y(column + ':Q', bin=True, title=None),
        color = alt.value('#1a7971')  
    ).properties(
        title=f'Распределение значений для {column}',
        height=300,
        width=300
    )

    combined_chart = alt.hconcat(boxplot_chart, hist_chart, spacing=30)  
    st.altair_chart(combined_chart)

