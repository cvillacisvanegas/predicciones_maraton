import pickle
from sklearn.pipeline import Pipeline
import os
import pandas as pd
from sklearn.model_selection import train_test_split
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
directorio_script = os.path.dirname(os.path.abspath(__file__))
nombre_archivo = os.path.join(directorio_script, 'produccion', 'ridge_model.pkl')
with open(nombre_archivo, 'rb') as model_file:
    loaded_model = pickle.load(model_file)
predictions = loaded_model.predict(datos_test)
results_df = pd.DataFrame({'Actual_MarathonTime': etiquetas_test, 'Predicted_MarathonTime': predictions})
results_df.to_csv('data/predict.csv', index=False)
print('se ha guardado el archivo "data/predict.csv" con las predicciones del grupo test')