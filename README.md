# CaptstoneProject

- Sandra Díaz
- Daniel Fares
- Oriol Soler

# 0. Summary (Oriol)

# 1. Introducción al problema (problema, como obtenemos los datos) (Oriol)

# 2. Data analysis
## 2.1. Descriptiva (Dani)
## 2.2. Disponibilidad de bicicletas (Dani)
## 2.3. Correlación entre variables (Sandra)
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


## 2.4. Capacidad y porcentaje de anclajes disponibles (Sandra)
Con la finalidad de entender el uso de las estaciones de bicicletas a lo largo del tiempo, se analiza en primer lugar el porcentaje de disponibilidad de bicicletas para cada estación (ctx0) por cada uno de los meses:
<img width="521" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/ca41940b-fade-4336-936e-ba3bec919c19">

Se observa que existen dos picos claros de uso: a primera hora de la mañana y a lo largo de la tarde, a lo largo de todos los meses. Como esto parece coincidir con el horario laboral, adicionalmente analizamos el uso por horas según días de la semana:
<img width="534" alt="image" src="https://github.com/or1ol/CaptstoneProject/assets/116820348/a0350806-26a9-4f7b-bf57-e9fa053b1a00">

El patrón detectado anteriormente coincide con los días de actividad profesional (lunes-viernes), y pierde relevancia el fin de semana (sábado y domingo). Es por eso que decidimos, más adelante, generar una nueva variable de días festivos (ver apartado 4).

## 2.5. Key Insights (Dani) (Sandra)

# 3. Data cleaning

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
