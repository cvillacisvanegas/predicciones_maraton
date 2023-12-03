import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from datetime import datetime
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
df = pd.read_csv('data/MarathonData.csv')
df
df['Name']
df.info()
df['Wall21'] = pd.to_numeric(df['Wall21'],errors='coerce')
df.describe()
df = df.drop(columns=['Name'])
df = df.drop(columns=['id'])
df = df.drop(columns=['Marathon'])
df = df.drop(columns=['CATEGORY'])
df
df.isna().sum()
df = df.dropna(subset=['Wall21'])
df['CrossTraining'] = df['CrossTraining'].fillna(0)
df
df['CrossTraining'].unique()
valores_cross = {'CrossTraining':{'ciclista 1h':1, 'ciclista 3h':2, 'ciclista 4h':3, 'ciclista 5h':4, 'ciclista 13h':5}}
df.replace(valores_cross, inplace=True)
df
df['Category'] = df['Category'].fillna(0)
df
df['Category'].unique()
valores_category = {'Category':{'MAM':1, 'M45':2, 'M40':3, 'M50':4, 'M55':5, 'WAM':6}}
df.replace (valores_category, inplace=True)
df
df = df.query('sp4week<1000')
df
datos_train = df.sample(frac=0.8,random_state=0)
datos_test = df.drop(datos_train.index)
etiquetas_train = datos_train.pop('MarathonTime')
etiquetas_test = datos_test.pop('MarathonTime')
datos_train
datos_train, datos_test, etiquetas_train, etiquetas_test = train_test_split(
    df[['Category', 'km4week', 'sp4week', 'CrossTraining', 'Wall21']], df['MarathonTime'], test_size=0.2, random_state=42
)
pipelines = {
    'LinearRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ]),
    'Lasso': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ]),
}
param_grids = {
    'LinearRegression': {},
    'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
    'Lasso': {'model__alpha': [0.1, 1.0, 10.0]},
    'DecisionTree': {'model__max_depth': [None, 10, 20, 30]},
    'RandomForest': {'model__n_estimators': [10, 50, 100]},
    'SVR': {'model__C': [0.1, 1.0, 10.0], 'model__kernel': ['linear', 'rbf']},
    'NeuralNetwork': {'model__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'model__alpha': [0.0001, 0.001, 0.01]}
}
for model_name, pipeline in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, scoring='r2')
    grid_search.fit(datos_train, etiquetas_train)
    train_score = grid_search.best_estimator_.score(datos_train, etiquetas_train)
    test_score = grid_search.best_estimator_.score(datos_test, etiquetas_test)
    print(f"Modelo: {model_name}")
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"Puntuación en Train: {train_score}")
    print(f"Puntuación en Test: {test_score}")
    print("-" * 50)
ridge_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))  
])
ridge_model.fit(datos_train, etiquetas_train)
fecha_entrenamiento = datetime.now().strftime("%y%m%d%H%M%S")
nombre_archivo = f'model_{fecha_entrenamiento}.pkl'
with open(nombre_archivo, 'wb') as model_file:
    pickle.dump(ridge_model, model_file)
with open(nombre_archivo, 'rb') as model_file:
    loaded_model = pickle.load(model_file)
predictions = loaded_model.predict(datos_test)
print("El modelo ha sido entrenado y guardado.")

