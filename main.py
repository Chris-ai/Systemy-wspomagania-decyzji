from pandas.core import base
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


def firstModule():
    st.subheader('Moduł 1')
    
    uploaded_file = st.file_uploader("", type="xlsx")

    if uploaded_file:
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv('data/income.csv', sep =';', decimal=',')
    
    st.markdown('---')
    st.subheader('Zamiana danych tekstowych na numeryczne')
    
    classColumn = st.selectbox(
        'Wybór kolumny decyzyjnej',
        data.columns.values,
        index= len(data.columns.values) - 1 
    )
    
    colLeft, colRight = st.columns(2)
    
    with colLeft: 
        textColumns = st.multiselect(
            'Wybierz kolumny',
            data.columns.values
        )
    with colRight:
        typeOfColumn = st.selectbox(
            'Wybierz sposób zamiany danych',
            ['Alfabetycznie', 'Kolejność występowania']
        )
    
    dataModified = data.copy()
    
    for col in textColumns:
        if typeOfColumn == 'Alfabetycznie':
            dataModified[col + '_alfabetycznie'] = pd.Categorical(dataModified[col]).codes
        elif typeOfColumn == 'Kolejność występowania':
            dataModified[col + '_kolejnoscWystepowania'] = pd.factorize(dataModified[col])[0]
            
    st.dataframe(dataModified)
    
    st.markdown('---')
    st.subheader('Dyskretyzacja zmiennych rzeczywistych na określoną liczbę przedziałów')
    
    leftColumn, rightColumn = st.columns(2)
    
    with leftColumn:         
        rangesColumn = st.multiselect(
            'Wybierz kolumny ',
            data.columns.values
        )
        
    with rightColumn:
        q = st.number_input(
            'Ilość przedziałów', step=1, value=1
        )
    
    for col in rangesColumn:
        dataModified[col + '_przedzial'] = pd.qcut(dataModified[col], q=q, duplicates='drop')
        
    st.dataframe(dataModified)
    
    st.markdown('---')
    st.subheader('Normalizacja zmiennych')
       
    normsColumn = st.multiselect(
            'Wybierz kolumny   ',
            dataModified.columns.values
        )
    
    for col in normsColumn:
        dataModified[col + '_norm'] = ( ((dataModified[col])  - (dataModified[col]).mean())/dataModified[col].std())

    st.dataframe(dataModified)
    
        
    st.markdown('---')
    st.subheader('Zmiana przedziału wartości z oryginalnego <min; max> ')
    
    columns = st.multiselect(
        'Kolumny',
        dataModified.columns.values
    )
    
    leftNumberInput, rightNumberInput = st.columns(2)
    
    with leftNumberInput:
        a = st.number_input(
            'min', step = 5, value = -100
        )
    with rightNumberInput:
        b = st.number_input(
            'max', step= 5, value= 100
        )
    
    minMax = st.slider('Przedział', a, b,(a,b))
    
    min_max_scaler = MinMaxScaler(feature_range=(int(minMax[0]),int(minMax[1]))) 
    
    for col in columns:
        dataModified[col] = min_max_scaler.fit_transform(dataModified[[col]])
        
    st.dataframe(dataModified)
    
    st.markdown('---')
    colLeft, colCenter, colRight = st.columns(3)

    with colLeft:
        q = st.number_input(
            '                                   ',
            step = 10,
            value = 10
        )
    minMaxList = ['% Najmniejszych z', '% Największych z']
    with colCenter:
        minMax = st.selectbox(
            '                                                 ',
            minMaxList
        )

    with colRight:
        col = st.selectbox(
            '                                        ',
            dataModified.columns.values
        )

    numberOfRows = len(dataModified)
    percentage = q/100
    n = int(numberOfRows*percentage)

    if (minMax == minMaxList[0]):
        st.dataframe(dataModified.nsmallest(n, col))
        
    elif (minMax == minMaxList[1]):
        st.dataframe(dataModified.nlargest(n, col))

    st.markdown('---')
    st.subheader('Wykresy dla poszczególnych kolumn')
    
    colLeft, colRight = st.columns(2)

    with colLeft:
        x = st.selectbox(
        'X', 
        dataModified.columns.values
        
    )
    with colRight:
        y = st.selectbox(
        'Y', 
        dataModified.columns.values
    )

    fig = px.scatter(
        data,
        x=x,
        y=y,
        color = classColumn
    )

    st.plotly_chart(fig)
    
    
    st.markdown('---')
    st.subheader('Histogramy')

    colLeft, colRight = st.columns(2)

    with colLeft:
        xHistogram = st.selectbox(
        'Dla której kolumny chcesz narysować histogram?', 
        dataModified.columns.values
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
             dataModified.columns.values
        )

    with colCenter:
        y3D = st.selectbox(
            'Zmienna y dla wykresu 3D',
            dataModified.columns.values 
        )

    with colRight:
        z3D = st.selectbox(
            'Zmienna z dla wykresu 3D',
            dataModified.columns.values
        )


    fig = px.scatter_3d(
        data,
        x = x3D,
        y = y3D,
        z = z3D,
        color = classColumn,
        symbol = classColumn
    )

    st.plotly_chart(fig)
    
def secondModule():
    st.subheader('Moduł 2')

def main():
    st.set_page_config(page_title = 'Systemy wspomagania decyzji')
    st.header('SWD 2021')
    
    choice = st.sidebar.selectbox(
        '',
        ['Moduł 1', 'Moduł 2']
    )

    if choice == 'Moduł 1':
        firstModule()
    elif choice == 'Moduł 2':
        secondModule()
        
main()