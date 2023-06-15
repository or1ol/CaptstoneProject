# CaptstoneProject

- Sandra Díaz
- Daniel Fares
- Oriol Soler

# 0. Summary (Oriol)

# 1. Introducción al problema (problema, como obtenemos los datos) (Oriol)
El reto propuesto para el proyecto Capstone consiste en predecir el porcentaje de sitios disponibles para aparcar las bicis de Bicing Barcelona por estación según sus datos históricos. Estos son recogidos y publicados mensualmente al portal Open Data del ayuntamiento de Barcelona, y contienen parametros relativos a cada estación y sus bicicletas.

Uno de los primeros problemas a afrontar es la gran cantidad de datos disponibles. Este hecho dificulta en gran medida las primeras etapas del proyecto dado que la obtención y primeros análisis de los datos llevan mucho tiempo y esfuerzo. Para conseguirlo, se usa Dask, una librería de Python que permite parallel computing.

Posteriormente, se procede a hacer un estudio detallado de cada dimensión, estudiar que correlaciones hay entre  las variables, limpiar los datasets, enriquecer los datos y procesarlos para crear modelos predictivos.

# 2. Data cleaning
- data set, descargado datos dek 2019, 2020, 2021, 2022, 2023 (datos de test)
- habia datos de 2018, 2019 (meses: 01, 02, 03)
- convertir datos de timestamp (drop duplicates)
- processing data por año
- 

# 3. Data analysis

## 3.1. Descriptiva (Dani)
un resumen de los que hemos visto de los notebook de exploring por año
y comparacion 
hablar del tema del covid

## 3.2. Disponibilidad de bicicletas (Dani)
Añadir imagenes de la data de la num_bikes_available por año 

## 3.3. Correlación entre variables (Sandra)
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

## 3.4. Capacidad y porcentaje de anclajes disponibles (Sandra)
Con la finalidad de entender el uso de las estaciones de bicicletas a lo largo del tiempo, se analiza en primer lugar el porcentaje de disponibilidad de bicicletas para cada estación (ctx0) por cada uno de los meses:
<img width="521" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/ca41940b-fade-4336-936e-ba3bec919c19">

Se observa que existen dos picos claros de uso: a primera hora de la mañana y a lo largo de la tarde, a lo largo de todos los meses. Como esto parece coincidir con el horario laboral, adicionalmente analizamos el uso por horas según días de la semana:
<img width="534" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/a0350806-26a9-4f7b-bf57-e9fa053b1a00">

El patrón detectado anteriormente coincide con los días de actividad profesional (lunes-viernes), y pierde relevancia el fin de semana (sábado y domingo). Es por eso que decidimos, más adelante, generar una nueva variable de días festivos (ver apartado 4).

## 3.5. Key Insights (Dani) (Sandra)

# 4. Data enirchment (festivos, meteorología)
## 4.1. Días festivos
En el análisis del punto 2 se ha detectado que los días que caen en fin de semana se asocian con cambios en la demanda de las bicicletas. Esto está directamente relacionado con que son días no laborales. Sin embargo, hay que tener en cuenta que los festivos locales y nacionales como la Diada o Navidad, en caso de caer en día laboral, no se están interpretando como no laborables. Para ello, a partir de una base de datos que indica los días laborales desde 2019 hasta 2023, se han generado las siguientes variables adicionales:

- Festius: marca los festivos locales, autonómicos y nacionales como tal.
- Festius_sun: adicionalmente a lo anterior, añade los domingos como festivos.
- Festius_sun_sat: además de lo anterior, se incluye el sábado como festivo.

# 5. Data processing

# 6. Data prediction (model comparison)

# 7. Results

# 8. Conclusions

# 9. Next steps, suggerencias

# 10. Anexos
url a los notebooks

![prueba insertar imagen](./img/img_prueba.jpeg)
