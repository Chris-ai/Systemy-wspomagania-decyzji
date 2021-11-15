import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title = 'Systemy wspomagania decyzji')
st.header('SWD 2021')

choice = st.sidebar.selectbox(
    "Wybierz moduł",
    ['Moduł 1', 'Moduł 2']
)

# MODUŁ 1 #
if choice == 'Moduł 1':
    st.subheader('Moduł 1')

    data = pd.read_csv('INCOME.TXT', sep ='\t', decimal=',')
    allColumns = ['Aktywa','Przych','Aktywa_przedzial','Przych_przedzial','Aktywa_norm','Przych_norm','Hrabstwo','Hrabstwo_num_kol_wystp','Hrabstwo_num_alf']

    columns = st.multiselect(
        'Jakie kolumny mają być wyświetlone?',
        allColumns
    )

    number = st.number_input('Na ile przedziałów chcesz podzielić zbiór?', step=1, value=1)

    # PRZEDZIALY DANYCH #
    data['Aktywa_przedzial'] = pd.qcut(data['Aktywa'], q=number, duplicates='drop')
    data['Przych_przedzial'] = pd.qcut(data['Przych'], q=number, duplicates='drop')

    # KOLUMNY ZE ZMIENNYMI TEKSTOWYMI NA LICZBOWE #
    data['Hrabstwo_num_kol_wystp'] = pd.factorize(data['Hrabstwo'])[0] #kolejnosc występowania

    data['Hrabstwo_num_alf'] = pd.Categorical(data['Hrabstwo']).codes #alfabetycznie

    # NORMALIZACJA #
    data['Aktywa_norm'] = ( ((data['Aktywa'])  - (data['Aktywa']).mean())/data['Aktywa'].std())
    data['Przych_norm'] = ( ((data['Przych'])  - (data['Przych']).mean())/data['Przych'].std())


    if len(columns) == 0:
        st.dataframe(data[allColumns])
    else:
        st.dataframe(data[columns])

    st.markdown('---')
    st.subheader('Zmiana przedziału min, max')

    data = data[allColumns]


    minMaxCol = st.selectbox(
        'Wybierz kolumnę',
        ['Aktywa','Przych']
    )

    minMax = st.slider('Przedział', -100, 100,(-100,100))

    min_max_scaler = MinMaxScaler(feature_range=(int(minMax[0]),int(minMax[1]))) 
    data[minMaxCol] = min_max_scaler.fit_transform(data[[minMaxCol]])

    st.dataframe(data.drop(['Przych_przedzial', 'Aktywa_przedzial','Hrabstwo_num_kol_wystp','Hrabstwo_num_alf'], axis=1))

    st.markdown('---')

    colLeft, colCenter, colRight = st.columns(3)

    with colLeft:
        q = st.number_input(
            '',
            step = 10,
            value = 10
        )
    minMaxList = ['% Najmniejszych z', '% Największych z']
    with colCenter:
        minMax = st.selectbox(
            '',
            minMaxList
        )

    with colRight:
        col = st.selectbox(
            '',
            ['Aktywa','Przych','Aktywa_norm','Przych_norm']
        )

    numberOfRows = len(data)
    percentage = q/100
    n = int(numberOfRows*percentage)

    if (minMax == minMaxList[0]):
        st.dataframe(data.drop(['Przych_przedzial', 'Aktywa_przedzial','Hrabstwo_num_kol_wystp','Hrabstwo_num_alf'], axis=1).nsmallest(n, col))
        
    elif (minMax == minMaxList[1]):
        st.dataframe(data.drop(['Przych_przedzial', 'Aktywa_przedzial','Hrabstwo_num_kol_wystp','Hrabstwo_num_alf'], axis=1).nlargest(n, col))


    st.markdown('---')
    st.subheader('Wykresy dla poszczególnych kolumn')

    colLeft, colRight = st.columns(2)

    with colLeft:
        x = st.selectbox(
        'X', 
        ['Aktywa','Przych','Aktywa_norm','Przych_norm'],
        
    )
    with colRight:
        y = st.selectbox(
        'Y', 
        ['Przych','Aktywa','Aktywa_norm','Przych_norm']
    )

    fig = px.scatter(
        data,
        x=x,
        y=y,
        color='Hrabstwo'
    )

    st.plotly_chart(fig)


    st.markdown('---')
    st.subheader('Histogramy')


    colLeft, colRight = st.columns(2)

    with colLeft:
        xHistogram = st.selectbox(
        'Dla której kolumny chcesz narysować histogram?', 
        ['Aktywa','Przych','Aktywa_norm','Przych_norm','Hrabstwo']
    )  

    with colRight:
        bins = st.number_input('Ile "binów" ?', step=5, value=0)


    fig = px.histogram(
        data,
        x = xHistogram,
        nbins = bins
    )

    st.plotly_chart(fig)

    st.markdown('---')
    st.subheader('Wykres 3D')

    colLeft, colCenter, colRight = st.columns(3)

    with colLeft:
        x3D = st.selectbox(
            'Zmienna x dla wykresu 3D',
            ['Aktywa','Przych','Aktywa_norm','Przych_norm', 'Hrabstwo_num_kol_wystp','Hrabstwo_num_alf']
        )

    with colCenter:
        y3D = st.selectbox(
            'Zmienna y dla wykresu 3D',
            ['Przych','Aktywa','Aktywa_norm','Przych_norm', 'Hrabstwo_num_kol_wystp','Hrabstwo_num_alf']
        )

    with colRight:
        z3D = st.selectbox(
            'Zmienna z dla wykresu 3D',
            ['Hrabstwo_num_kol_wystp','Hrabstwo_num_alf','Przych','Aktywa','Aktywa_norm','Przych_norm']
        )


    fig = px.scatter_3d(
        data,
        x = x3D,
        y = y3D,
        z = z3D,
        color = 'Hrabstwo',
        symbol = 'Hrabstwo'
    )

    st.plotly_chart(fig)
    
elif choice == "Moduł 2":
    st.subheader("Moduł 2")
    
    uploaded_file = st.file_uploader("Załącz plik typu XLSX", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        st.dataframe(df)
        st.write(type(uploaded_file))