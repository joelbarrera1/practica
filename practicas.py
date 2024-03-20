###prácticas
#### modelo de machine lerning
#### cargamos las librerias necesarias
import pandas as pd
import numpy as ny
import matplotlib.pyplot as mat
import seaborn as sea
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

####cargamos los datos, los pasamos a logaritmos y los visualizamos
basedatos = '/Users/joelbarreraperez/Desktop/practicas/Base_de_datos.xlsx'
df = pd.read_excel(basedatos,sheet_name='datos')
df = df.drop(columns=['Año'])
numeric_columns = ['Inflacion', 'TasaInteres', 'TDC ', 'PIBPC', 'INPC ']
df[numeric_columns] = ny.log1p(df[numeric_columns])
print(df)

#### Analisis de los datos, visualizamos los histogramas y el boxplot###
print(df.describe())
###histograma
df.hist(bins=20, figsize=(15, 10))
mat.suptitle('Histogramas de las Variables(log)')
mat.show()
# Boxplot para observar la distribución de las variables
mat.figure(figsize=(15, 8))
sea.boxplot(data=df[numeric_columns])
mat.title('Boxplot de las Variables (log)')
mat.show()

###Seleccionamos las variables de control (x) y la variable dependiente(y)###
#####omitimos la variable Inflacion por temas de correlacion con el INPC###
x = df[['TasaInteres', 'TDC ', 'INPC ']]
y = df['PIBPC']

###Dividimos los datos de entrenamiento y de prueba##
x_entr, x_prueba, y_entr, y_prueba = train_test_split(x, y, test_size=0.4, random_state=50)

####creamos el modelo de regresión lineal y lo entrenamos###
model = LinearRegression()
model.fit(x_entr, y_entr)

###Realizamos las predicciones en el conjunto de prueba###
predicciones = model.predict(x_prueba)
print(predicciones)

####Revisamos las estadisticas que nos ayudaran a comprobar el rendimiento del modelo##
eam = mean_absolute_error(y_prueba, predicciones)
ecm = mean_squared_error(y_prueba, predicciones)
rec = ny.sqrt(mean_squared_error(y_prueba, predicciones))
r_cuadrado = r2_score(y_prueba, predicciones)
print('Error absoluto medio:', eam)
print('Error cuadrático medio:', ecm)
print('Raíz del error cuadrático medio:', rec)
print('r2', r_cuadrado)

###Graficamos las predicciones con los valores reales##
mat.scatter(y_prueba, predicciones, alpha=0.5)
mat.title('Predicciones vs Valores Reales')
mat.xlabel('Valores Reales')
mat.ylabel('Predicciones')
mat.plot([min(y_prueba), max(y_prueba)], [min(y_prueba), max(y_prueba)], linestyle='-', color='blue', linewidth=3)
mat.show()
