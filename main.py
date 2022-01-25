from types import CellType
from matplotlib import markers
from pandas.core import base
from pandas.io.formats.format import buffer_put_lines
from sklearn import metrics
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from random import seed
from random import randrange
from math import dist, sqrt
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.metrics import matthews_corrcoef

from streamlit.elements.arrow import Data
from streamlit.elements.color_picker import ColorPickerMixin

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
        dataModified,
        x = xHistogram,
        nbins = bins,
        color = classColumn
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

# Euclidean distance
def euclideanDistance(data, newObj):
    distance = []
    for index, row in data.iloc[:,:-1].iterrows():
        sum = 0.0
        for i in range(len(newObj)):
            # print(row[i], newObj[i])
            sum += (np.float32(row[i]) - np.float32(newObj[i]))**2
        
        distance.append(sqrt(sum))
    
    return distance

# Manhattan distance
def manhattanDistance(data, newObj):
    distance = []

    for index, row in data.iloc[:,:-1].iterrows():
        sum = 0.0
        for i in range(len(newObj)):
            sum += abs(row[i] - newObj[i])
            
        distance.append(sum)
        
    return distance

# Chebyshev distance
def chebyshevDistance(data, newObj):
    distance = []

    for index, row in data.iloc[:,:-1].iterrows():
        listOfDistances = []
        for i in range(len(newObj)):
            listOfDistances.append(row[i]-newObj[i])
            maxDistance = max(listOfDistances)
        distance.append(maxDistance)

    return distance

# Mahalanobis distance
def mahalanobisDistance(data, newObj):
    distance = []
    v_m = []
    data = data.iloc[:,:-1]
    cov_data = data.cov()
    print(cov_data)
    
    for index, row in data.iterrows():
        v_m = []
        for i in range(len(newObj)):
            v_m.append(newObj[i] - row[i])
        
        result = np.array(v_m)
        #result = np.dot(result, np.linalg.inv(cov_data))
        result = np.dot(result,cov_data)
        result = np.dot(result, v_m)

        distance.append(np.sqrt(result))
    
    return distance

# get k neighbors
def getNeighbors(data, metric, k):
    neighbors = []
    # specify column to sort and get ksmallest distances
    if metric == 'Euklidesowa':
        neighbors = data.sort_values(by='Metryka Euklidesowa').head(k)
        data = data.drop(['Metryka Euklidesowa'], axis = 1)
        
    elif metric == 'Manhattan':
        neighbors =  data.sort_values(by='Metryka Manhattan').head(k)
        data = data.drop(['Metryka Manhattan'], axis = 1)
        
    elif metric == 'Czebyszewa':
        neighbors =data.sort_values(by='Metryka Chebysheva').head(k)
        data = data.drop(['Metryka Chebysheva'], axis = 1)
        
    elif metric == 'Mahalanobisa':
        neighbors = data.sort_values(by='Metryka Mahalanobisa').head(k)
        data = data.drop(['Metryka Mahalanobisa'], axis = 1)
    
    return neighbors, data

def resolveConflicts(data,neighbors,k,classColumn,potentialConflicts, metric,newObject):
    
    
    potentialConflictsSums = []
    minSumsOfPotentialConflicts = []
    
    for n in potentialConflicts:
        sum = 0.0
        sum = neighbors.loc[neighbors[classColumn] == n].iloc[:,-1:].sum()
        potentialConflictsSums.append([n, sum[0]])

    minVal = potentialConflictsSums[0][1]
    
    # Searching for the minimum sum 
    for name, sumVal in potentialConflictsSums:
        if sumVal < minVal:
            minVal = sumVal
    
    # If sum == minimum we assign the classs
    for name, sumVal in potentialConflictsSums:
        if sumVal == minVal:
            minSumsOfPotentialConflicts.append([name, sumVal])
    print(potentialConflictsSums)

    print(minSumsOfPotentialConflicts)
    # Checking if there is more than one minimum sum of distances
    if (len(minSumsOfPotentialConflicts) > 1): # If so, KNN for k+1
        predict = KNN(data,newObject,metric,k+1,classColumn)
    elif (len(minSumsOfPotentialConflicts) == 1 ): # If not, we return a prediction
        predict = minSumsOfPotentialConflicts[0][0]
        
    return predict

# Predykcja
def predictClassification(data, neighbors, k, classColumn, metric, newObject):
    df_value_counts = neighbors[classColumn].value_counts().reset_index()
    df_value_counts.columns = ['name', 'counts']
    maxVal = df_value_counts['counts'].max()
    
    potentialConflicts = []
    classOfNewObj = None

    for index, row in df_value_counts.iterrows():
        if(row['counts'] == maxVal):
            potentialConflicts.append(row['name'])

    if (len(potentialConflicts) > 1):
        classOfNewObj = resolveConflicts(data, neighbors, k, classColumn,potentialConflicts, metric, newObject)
    elif (len(potentialConflicts) == 1):
        classOfNewObj = potentialConflicts[0]
        
    return classOfNewObj
     
# Alogrithm
def KNN(data, newObject, metric, k, classColumn):
    # 1st step - distance between objects - euclidean/manhattan/chebyshev/mahalanobisDistance
    # 2nd step - sort data by distances                - getNeighbors()
    # 3rd step - get K objects with smallest distances - getNeighbors()
    # 4th step - resolve conflicts - predictClassification() -> resolveConflicts()
    
    if metric == 'Euklidesowa':
        data['Metryka Euklidesowa']  = euclideanDistance(data, newObject)
        
    elif metric == 'Manhattan':
        data['Metryka Manhattan'] = manhattanDistance(data, newObject)
        
    elif metric == 'Czebyszewa':
        data['Metryka Chebysheva'] = chebyshevDistance(data, newObject)
        
    elif metric == 'Mahalanobisa':
        data['Metryka Mahalanobisa'] = mahalanobisDistance(data, newObject)

    neighbors, data = getNeighbors(data, metric, k)
    
    prediction = predictClassification(data, neighbors, k, classColumn, metric, newObject)
    # print('pred: ', prediction)
    # print('----')

    # prediction = np.argmax(prediction)
    # newObject.append(prediction)
    # st.write(newObject)
    return prediction

def createNewObj(row, size):
    newObj = []
    for n in range(size):
        newObj.append(row[n])

    return newObj

def secondModule():
    st.subheader('Moduł 2')
    uploaded_file = st.file_uploader("", type="xlsx")

    if uploaded_file:
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv('data/income.csv', sep =';', decimal=',')
    
    
    st.dataframe(data)
    st.markdown('---')
    
    modes = ['Klasyfikacja', 'Ocena jakości klasyfikacji']
    metrics = ['Euklidesowa', 'Manhattan', 'Czebyszewa', 'Mahalanobisa']
    mode = st.selectbox(
        'Wybierz tryb modułu',
        modes
    )
    
    classColumn = st.selectbox(
        'Kolumna decyzyjna:',
        data.columns.values,
        index= len(data.columns.values) - 1 
    )
    
    

    if mode == modes[0]: #Klasyfikacja
       
       leftColumn, rightColumn =  st.columns(2)
       newObject = []
       
       with leftColumn: 
            for col in data.columns[:-1]:
                n = st.number_input(
                    col,
                    step = 1.0,
                    format="%.2f"
                )
                newObject.append(n)     
       with rightColumn:
            metric = st.selectbox(
                'Wybierz metrykę',
                metrics
            )
            
            k = st.number_input(
                'Liczba sąsiadów',
                value=3,
                step=1
            ) 
            
            if st.button('Klasyfikuj obiekt'):
                pred = KNN(data, newObject, metric, k, classColumn)
                st.write(pred)

    elif mode == modes[1]: #Ocena jakości klasyfikacji
        kColumn, metricColumn = st.columns(2)
        
        with kColumn:
            k = st.number_input(
                'Liczba sąsiadów',
                value=3,
                step=1
            ) 
            
        with metricColumn:
             metric = st.selectbox(
                'Wybierz metrykę',
                metrics
            )
             
        data[classColumn] = pd.Categorical(data[classColumn]).codes     
        dataModified = data.copy()
        
        # for col in data.columns.drop([classColumn]):
        #     data[col] = pd.to_numeric(data[col], downcast="float")
            
        dataNormalized = data.copy()
        for col in dataNormalized.columns.drop([classColumn]):
            
            dataNormalized[col] = ( ((dataNormalized[col])  - (dataNormalized[col]).mean())/dataNormalized[col].std())
            
        
        if st.button('Ocena jakości'):
            for index, row in data.iterrows():
                newObj = createNewObj(row,len(data.iloc[:,:-1].columns))
                dataModified.loc[index, classColumn] =  KNN(dataModified.iloc[index:],newObj,metric,k,classColumn)
                # dataModified.loc[index, classColumn] = pred
        originalCol, modifiedCol = st.columns(2)
            
        with originalCol:
            st.dataframe(data)
        with modifiedCol:  
            st.dataframe(dataNormalized)
            
        col1, col2, col3 = st.columns(3)
        col1.write('Dokładność algorytmu: ')
        col2.write(round(accuracy_score(data[classColumn],dataModified[classColumn])*100, 2))
        col3.write('%')

# RAPORT #            

        # file = open('data/data-irisNormalized-quality-assessment.txt', 'w')   
        # if st.button('Ocena jakości'):
            
        #     for metric in metrics:
        #         for k in range(1, len(dataNormalized)):
        #             dataIterrate = dataNormalized.copy()
        #             for index, row in dataIterrate.iterrows():
        #                 newObj = createNewObj(row,len(dataIterrate.iloc[:,:-1].columns))
        #                 dataIterrate.loc[index, classColumn] =  KNN(dataIterrate.iloc[index:],newObj,metric,k,classColumn)
                        
        #             print(k, round(accuracy_score(dataNormalized[classColumn],dataIterrate[classColumn])*100, 2), metric, file=file)
        #     file.close()  

def euclidean(pointA, pointB):
    pd.to_numeric(pointA)
    pd.to_numeric(pointB)
    return np.sqrt((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2)

def l1(pointA, pointB):
    return np.abs(pointA[0] - pointB[0]) + np.abs(pointA[1] - pointB[1])
    
def lInfinity(pointA, pointB):
    return  max(abs((pointA[0] - pointB[0])), abs((pointA[1] - pointB[1])))

def mahalanobis(pointA, pointB):
    v_m = []
    cov_data = np.cov(pointA)
    
    for i in range(len(pointB)):
        v_m.append(pointB[i] - pointA[i])
        print(v_m, 'spacja ', cov_data)
        result = np.array(v_m)
        # result = np.dot(result, np.linalg.inv(cov_data))
        result = np.dot(result,cov_data)
        result = np.dot(result, v_m)
        
    return np.sqrt(result)

def kMeans(data, k, metric):
    diff = 1
    cluster = np.zeros(data.shape[0])
    
    #select k random centroirds
    random_id = np.random.choice(len(data), size = k, replace = False)
    centroids = data.iloc[random_id, :].values
    
    while diff:
        
        for i,row in enumerate(data.values):
            mn_dist = float('inf')
            for idx, centroid in enumerate(centroids): 
                if metric == 'Euklidesowa':
                    d = euclidean(centroid, row)
                elif metric == 'l1':
                    d = l1(centroid, row)
                elif metric == 'l nieskończoność':
                    d = lInfinity(centroid, row)
                elif metric == 'Mahalanobisa':
                    d = mahalanobis(centroid, row)
                    
                print(mn_dist, 'd: ', d)
                
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx
                
                print(cluster)
        
               
        new_centroids = pd.DataFrame(data).groupby(by=cluster).mean().values
        print(new_centroids, 'soacjaaaa ' ,centroids, 'd= ', d) 
        if np.count_nonzero(centroids-new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids        
        
    return centroids, cluster

def calculate_cost(X, centroids, cluster):
    sum = 0
    for i, val in enumerate(X):
        sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 + (centroids[int(cluster[i]), 1]-val[1])**2)
            
    return sum

def thirdModule():
    st.subheader('Moduł 3')
    uploaded_file = st.file_uploader("", type=["csv", 'txt'])
    col1, col2 = st.columns(2)
    metrics = ['Euklidesowa', 'l1', 'l nieskończoność', 'Mahalanobisa']
    with col1:  
        header_radio = st.radio(
        "Dane",
        ('Z nagłówkami', 'Bez nagłówków'))
        
    with col2:
        class_column = st.radio(
        "",
        ('Z klasą decyzyjną', 'Bez klasy decyzyjnej'))
          
    
    if uploaded_file:
        if header_radio == 'Bez nagłówków':
            data = pd.read_csv(uploaded_file, sep ="\s+|;|:", decimal=',', header=None) 
        elif header_radio == 'Z nagłówkami':
            data = pd.read_csv(uploaded_file, sep ="\s+|;|:", decimal=',')
            
            whole_data = data.copy()
            
        if class_column == 'Z klasą decyzyjną':
                data = data.drop(data.iloc[:,-1:], axis=1)
    else:
        if header_radio == 'Bez nagłówków':
            data = pd.read_excel('data/income.xlsx', header=None)
        elif header_radio == 'Z nagłówkami':
            data = pd.read_excel('data/income.xlsx')
            
            whole_data = data.copy()
            
        if class_column == 'Z klasą decyzyjną':
            data = data.drop(data.iloc[:,-1:], axis=1)

    st.dataframe(data)
    st.markdown('---')
    st.subheader('Algorytm K-średnich')
    
    colLeft, colCenter, colRight = st.columns(3)
    
    with colLeft:
        k = st.number_input(
            "k",
            len(whole_data.iloc[:,-1:].value_counts()),
        )
    with colCenter:
        metric = st.selectbox(
                'Wybierz metrykę',
                metrics
            )
        
    with colRight:
        st.write('Grupowanie')
        button = st.button('Start')
        
        st.write('Elbow Method')
        buttonElbow = st.button("OK")
    
    if button:
        centroids, cluster = kMeans(data,k,metric)
        print(centroids, cluster)
        # print(whole_data.iloc[:,-1:].head(15))
    
    if buttonElbow:
        # Elbow Method #
  
        cost_list = []

        for k in range(1, 10):
            centroids, cluster = kMeans(data, k, metric)
            sum = 0
            for i, val in enumerate(data.values):
                cost = euclidean(centroids[int(cluster[i])], val)
                sum += cost
            
            cost_list.append(sum)
           

        fig = plt.figure(figsize=(15,15))
        print(cost_list, len(cost_list))
        sns.lineplot(x=range(1,10), y=cost_list, marker='o')
        plt.xlabel('k')
        plt.ylabel('WCSS')
        plt.savefig('elbowMethod-plt.png')
    
    # Acc_score #
    y_true_col = pd.factorize(whole_data[whole_data.columns.values[-1]])[0]
    
    y_true_alf = pd.Categorical(whole_data[whole_data.columns.values[-1]]).codes
    
    print(y_true_alf)
    print(y_true_col)
    
    print('Dokładność predykcji (Jaccard_score): ', max(jaccard_score(y_true_col, cluster, average='macro'),jaccard_score(y_true_alf, cluster, average='macro')) )
    
    print('Dokładność predykcji (Matthews_corrcoef): ', max(matthews_corrcoef(y_true_col, cluster),matthews_corrcoef(y_true_alf, cluster)))
    
    plt.figure(figsize=(15,15))
        
    fig = px.scatter(
        data,
        x=data.iloc[:,0],
        y=data.iloc[:, 1],
        color = cluster,
        symbol = cluster
    )
    fig.update_layout(showlegend=False)
        
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title = 'Systemy wspomagania decyzji')
    st.header('SWD 2021')
    
    choice = st.sidebar.selectbox(
        '',
        ['Moduł 1', 'Moduł 2', 'Moduł 3']
    )

    if choice == 'Moduł 1':
        firstModule()
    elif choice == 'Moduł 2':
        secondModule()
    elif choice == 'Moduł 3':
        thirdModule()
        
main()