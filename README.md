# CaptstoneProject

- Sandra Díaz
- Daniel Fares
- Oriol Soler

# 0. Summary

# 1. Introducción al problema
El reto propuesto para el proyecto Capstone consiste en predecir el porcentaje de sitios disponibles para aparcar las bicis de Bicing Barcelona por estación según sus datos históricos. Estos son recogidos y publicados mensualmente al portal Open Data del ayuntamiento de Barcelona, y contienen parametros relativos a cada estación y sus bicicletas.

Uno de los primeros problemas a afrontar es la gran cantidad de datos disponibles. Este hecho dificulta en gran medida las primeras etapas del proyecto dado que la obtención y primeros análisis de los datos llevan mucho tiempo y esfuerzo. Para conseguirlo, se usa Dask, una librería de Python que permite parallel computing.

Posteriormente, se procede a hacer un estudio detallado de cada dimensión, estudiar que correlaciones hay entre  las variables, limpiar los datasets, enriquecer los datos y procesarlos para crear modelos predictivos.

# 2. Data cleaning
En primer lugar, se ha realizado un análisis inicial con el objetivo de limpiar los ficheros de datos que carecen de sentido por algún motivo. Para ello, se han realizado los siguientes pasos:
- Descarga de los datos de la página web oficial para los años 2019, 2020, 2021, 2022 y 2023 (incluyendo los datos no incorporados en la descarga inicial).
- Cálculo de los valores faltantes (NaN) para las distintas variables de los datasets.
- Cálculo de los valores que se corresponden con 0 de las distintas variables de los datasets.
- Clasificación de las variables según si son categóricas o numéricas, así como el cálculo de valores únicos que devuelve cada una de ellas. 
- Eliminación de elementos duplicados de las variables en las que no tienen sentido, como por ejemplo el ‘last_reported’.
- Eliminación de columnas que no son necesarias: 'year_last_updated_date', 'month_last_updated_date', 'week_last_updated_date', 'dayofweek_last_updated_date', 'dayofmonth_last_updated_date', 'dayofyear_last_updated_date', 'hour_last_updated_date' y 'minutes_last_updated_date'.
- Ajuste de la variable ‘post_code’ al ser incorrecta.
- Ajuste variable ‘status’, agrupando bajo el valor 0 ‘in_service’ y bajo 1, ‘closed’.
- Crear nuevas columnas para el ‘last_reported’ y el ‘last_updated’, asignando nuevas variables a los valores devueltos.
- Uniformar el formato de los timestamp a fecha/hora.
- Agrupar el timestamp en múltiplos de 60 para poder reducir la base de datos trabajada, ya que nos interesaba tener los datos en granularidad por hora en vez de minuto.
- Incorporación de la variable ctx0, que relaciona los anclajes disponibles (num_docs_available) entre la capacidad (capacity), es decir, el número máximo de anclajes por estación. Adicionalmente, se incluyen también ctx1, ctx2, ctx3 y ctx4, que hacen referencia a la disponibilidad porcentual de bicicletas la hora anterior, las dos horas anteriores... y así sucesivamente. 

# 3. Data analysis

## 3.1. Descriptiva
un resumen de los que hemos visto de los notebook de exploring por año
y comparacion 
hablar del tema del covid

### 3.1.1. Station_ID
En primer lugar, se han analizado los IDs de estaciones a lo largo de los años, para verificar si variaban en términos de volumen. Se ha percibido que no todos los IDs son constantes a lo largo de los años, y es por eso que se localizan los IDs únicos que están presentes en todos los años, encontrando un total de 405. 

![image](https://github.com/or1ol/CaptstoneProject/assets/116820348/73a5d118-da37-4131-b03c-06d9b22a2291)

### 3.1.2. Anclajes disponibles (num_docks_available) (Oriol)

### 3.1.3. Bicicletas disponibles (bikes_available -total y per tipus-) (Oriol)

### 3.1.4. Ctx0 (num_docs_available/capacity) (Sandra)

## 3.2. Correlación entre variables (Sandra)
El objetivo es explorar la si existe una asociación entre dos variables para establecer si existe de una relacional lineal. En ese sentido, se ha estudiado la correlación entre:

- El número de anclajes disponibles (num_docks_available) y el número de bicicletas disponibles (num_bikes_available)
<img width="635" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/d2773375-cc30-472f-a5c4-41376c6da726">

- El número de anclajes disponibles (num_docks_available) y el número de bicicletas manuales disponibles

- El número de anclajes disponibles (num_docks_available) y el número de bicicletas eléctricas disponibles

- El número de anclajes disponibles (num_docks_available) y la hora (hour)

- El número de anclajes disponibles (num_docks_available) y el día de la semana (dayofweek)
<img width="575" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/6f9af222-6398-4115-8ac9-01bf59868efd">

- El número de anclajes disponibles (num_docks_available) y la hora (hour)
<img width="538" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/fdc2e56d-f9f2-4ae9-8bda-33e8affc2df7">

- La capacidad (capacity) y el día de la semana (dayofweek)

- La capacidad (capacity) y la hora (hour)

- La capacidad (capacity) y el número de bicicletas disponibles (num_bikes_available)
<img width="547" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/0c49eed1-a189-498f-914a-aa10bc948513">

- El ctx0 (num_docs_available/capacity) y el día de la semana (dayofweek)
<img width="460" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/edf49541-8b01-4232-b347-3a5eb8d6c278">

- El ctx0 (num_docs_available/capacity) y la hora (hour)
<img width="408" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/a7272579-f2f2-470d-a3fe-d4c9df52e76b">

### Capacidad y porcentaje de anclajes disponibles (Sandra)
Con la finalidad de entender el uso de las estaciones de bicicletas a lo largo del tiempo, se analiza en primer lugar el porcentaje de disponibilidad de bicicletas para cada estación (ctx0) por cada uno de los meses:
<img width="521" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/ca41940b-fade-4336-936e-ba3bec919c19">

Se observa que existen dos picos claros de uso: a primera hora de la mañana y a lo largo de la tarde, a lo largo de todos los meses. Como esto parece coincidir con el horario laboral, adicionalmente analizamos el uso por horas según días de la semana:
<img width="534" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/a0350806-26a9-4f7b-bf57-e9fa053b1a00">

El patrón detectado anteriormente coincide con los días de actividad profesional (lunes-viernes), y pierde relevancia el fin de semana (sábado y domingo). Es por eso que decidimos, más adelante, generar una nueva variable de días festivos (ver apartado 4).

## 3.3. Key Insights
Teniendo en cuenta la exploración de datos realizada, las principales conclusión que extrapolamos son las siguientes:
- Se eliminan los datos de la época de Covid por ser anómalos.
- ...

# 4. Data enirchment (festivos, meteorología)
## 4.1. Días festivos
En el análisis del punto 2 se ha detectado que los días que caen en fin de semana se asocian con cambios en la demanda de las bicicletas. Esto está directamente relacionado con que son días no laborales. Sin embargo, hay que tener en cuenta que los festivos locales y nacionales como la Diada o Navidad, en caso de caer en día laboral, no se están interpretando como no laborables. Para ello, a partir de una base de datos que indica los días laborales desde 2019 hasta 2023, se han generado las siguientes variables adicionales:

- Festius: marca los festivos locales, autonómicos y nacionales como tal.
- Festius_sun: adicionalmente a lo anterior, añade los domingos como festivos.
- Festius_sun_sat: además de lo anterior, se incluye el sábado como festivo.

## 4.2. Meteorología (Oriol)
Una casuística que no estaba contemplada en el dataset inicial era la de la meteorología. Esta puede tener un alto impacto en el uso de las bicicletas en la ciudad, y por esto se incluye en el estudio.

Los datos encontrados abarcan todos los años en los que se analiza el uso de bicicletas, e incluyen muchas variables ya sean relacionadas con la temperatura, humedad, presión atmosférica, precipitación, viento o irradiación solar. Estudiando la correlación entre las variables y  entendiendo cuales de ellas podían tener mayor efecto, se decide proceder con las siguientes:

- Temperatura media diaria
- Precipitación acumulada diaria


# 5. Data processing

# 6. Data prediction (model comparison)

# 7. Results

# 8. Conclusions

# 9. Next steps, sugerencias

# 10. Anexos (url a los notebooks)
Los documentos trabajados son los siguientes:
- Análisis completo explotarorio de los datos de 2019:
- Análisis completo explotarorio de los datos de 2020:
- Análisis completo explotarorio de los datos de 2021:
- Análisis completo explotarorio de los datos de 2022:
- Análisis completo explotarorio de los datos de 2023:
- Documento de funciones utilizadas 'tools':
- Scripts XXX:
- Otros modelos ejecutado:
- Modelo final ejectuado:


![prueba insertar imagen](./img/img_prueba.jpeg)
