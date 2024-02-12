# insurance_services_optimization_through_machine_learning

La compañía de seguros Sure Tomorrow quiere resolver varias tareas con la ayuda de machine learning y te pide que evalúes esa posibilidad.

# Descripción

La compañía de seguros Sure Tomorrow quiere resolver varias tareas con la ayuda de machine learning y te pide que evalúes esa posibilidad.

- Tarea 1: encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.
- Tarea 2: predecir la probabilidad de que un nuevo cliente reciba una prestación del seguro. ¿Puede un modelo de predictivo funcionar mejor que un modelo dummy?
- Tarea 3: predecir el número de prestaciones de seguro que un nuevo cliente pueda recibir utilizando un modelo de regresión lineal.
- Tarea 4: proteger los datos personales de los clientes sin afectar al modelo del ejercicio anterior. Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento u ofuscación de datos. Pero los datos deben protegerse de tal manera que no se vea afectada la calidad de los modelos de machine learning. No es necesario elegir el mejor modelo, basta con demostrar que el algoritmo funciona correctamente.

# Tarea 1. Clientes similares

En el lenguaje de ML, es necesario desarrollar un procedimiento que devuelva los k vecinos más cercanos (objetos) para un objeto dado basándose en la distancia entre los objetos. Es posible que quieras revisar las siguientes lecciones (capítulo -> lección)- Distancia entre vectores -> Distancia euclidiana

Distancia entre vectores -> Distancia Manhattan
Para resolver la tarea, podemos probar diferentes métricas de distancia.

Escribe una función que devuelva los k vecinos más cercanos para un 𝑛𝑡ℎ
objeto basándose en una métrica de distancia especificada. A la hora de realizar esta tarea no debe tenerse en cuenta el número de prestaciones de seguro recibidas. Puedes utilizar una implementación ya existente del algoritmo kNN de scikit-learn (consulta el enlace) o tu propia implementación. Pruébalo para cuatro combinaciones de dos casos- Escalado

los datos no están escalados
los datos se escalan con el escalador MaxAbsScaler
Métricas de distancia
Euclidiana
Manhattan
Responde a estas preguntas:- ¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?- ¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?

Respuestas a las preguntas

¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?

Sí, el hecho de que los datos no estén escalados puede afectar al algoritmo kNN. Esto se manifiesta en cómo se calcula la distancia entre los puntos. Cuando las características no están en la misma escala, aquellas con valores más grandes dominarán la contribución a la distancia total. Esto puede llevar a que las distancias se vean distorsionadas y que las características con valores más grandes tengan un impacto desproporcionado en la clasificación de los vecinos más cercanos.

¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?

Los resultados al utilizar la métrica de distancia Manhattan tienden a ser similares independientemente del escalado de los datos. Esto se debe a que la distancia Manhattan calcula la distancia entre dos puntos sumando las diferencias absolutas entre las coordenadas de cada punto. A diferencia de la distancia euclidiana, que considera la longitud del vector entre dos puntos, la distancia Manhattan se enfoca en la distancia horizontal y vertical entre los puntos en un espacio bidimensional.

Tarea 2. ¿Es probable que el cliente reciba una prestación del seguro?
En términos de machine learning podemos considerarlo como una tarea de clasificación binaria.

Con el valor de insurance_benefits superior a cero como objetivo, evalúa si el enfoque de clasificación kNN puede funcionar mejor que el modelo dummy. Instrucciones:

Construye un clasificador basado en KNN y mide su calidad con la métrica F1 para k=1...10 tanto para los datos originales como para los escalados. Sería interesante observar cómo k puede influir en la métrica de evaluación y si el escalado de los datos provoca alguna diferencia. Puedes utilizar una implementación ya existente del algoritmo de clasificación kNN de scikit-learn (consulta el enlace) o tu propia implementación.- Construye un modelo dummy que, en este caso, es simplemente un modelo aleatorio. Debería devolver "1" con cierta probabilidad. Probemos el modelo con cuatro valores de probabilidad: 0, la probabilidad de pagar cualquier prestación del seguro, 0.5, 1. La probabilidad de pagar cualquier prestación del seguro puede definirse como:

𝑃{prestación de seguro recibida}=(número de clientes que han recibido alguna prestación de seguronúmero) / (total de clientes)

Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30.

¿Es probable que el cliente reciba una prestación del seguro?

El análisis de la probabilidad de recibir una prestación del seguro se realizó utilizando un modelo dummy aleatorio con diferentes probabilidades (0, probabilidad real de recibir una prestación del seguro, 0.5 y 1). Los resultados muestran que el F1-score varía dependiendo de la probabilidad utilizada en el modelo dummy, lo que sugiere que la probabilidad de recibir una prestación del seguro influye en la capacidad del modelo para predecir correctamente dichas prestaciones.

¿El enfoque de clasificación kNN puede funcionar mejor que el modelo dummy?

Para responder a esta pregunta, se construyó y evaluó un clasificador kNN con diferentes valores de k (1 a 10). La evaluación se realizó utilizando la métrica F1-score. Comparando los resultados del clasificador kNN con los del modelo dummy, podemos determinar si el enfoque de clasificación kNN es más efectivo para predecir la recepción de prestaciones del seguro en comparación con el modelo dummy.

Tarea 3. Regresión (con regresión lineal)
Con insurance_benefits como objetivo, evalúa cuál sería la RECM de un modelo de regresión lineal.

Construye tu propia implementación de regresión lineal. Para ello, recuerda cómo está formulada la solución de la tarea de regresión lineal en términos de LA. Comprueba la RECM tanto para los datos originales como para los escalados. ¿Puedes ver alguna diferencia en la RECM con respecto a estos dos casos?

Denotemos- 𝑋
: matriz de características; cada fila es un caso, cada columna es una característica, la primera columna está formada por unidades- 𝑦
— objetivo (un vector)- 𝑦̂
— objetivo estimado (un vector)- 𝑤
— vector de pesos La tarea de regresión lineal en el lenguaje de las matrices puede formularse así:
𝑦=𝑋𝑤
El objetivo de entrenamiento es entonces encontrar esa 𝑤
w que minimice la distancia L2 (ECM) entre 𝑋𝑤
y 𝑦
:

min𝑤𝑑2(𝑋𝑤,𝑦)ormin𝑤MSE(𝑋𝑤,𝑦)
Parece que hay una solución analítica para lo anteriormente expuesto:
𝑤=(𝑋𝑇𝑋)−1𝑋𝑇𝑦
La fórmula anterior puede servir para encontrar los pesos 𝑤
y estos últimos pueden utilizarse para calcular los valores predichos
𝑦̂ =𝑋𝑣𝑎𝑙𝑤
Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30. Utiliza la métrica RECM para evaluar el modelo.

Observaciones

Basándonos en los resultados obtenidos, observamos que el modelo de regresión lineal ajustado a los datos originales muestra un RMSE (Error Cuadrático Medio) más bajo de aproximadamente 0.34, lo que indica una mayor precisión en las predicciones en comparación con el modelo ajustado a los datos escalados, que tiene un RMSE de aproximadamente 0.74. Aunque el R2 (Coeficiente de Determinación) se mantiene constante en ambos casos alrededor de 0.66, sugiriendo una consistencia en la capacidad explicativa del modelo, la diferencia en el RMSE sugiere que la información de la escala original de las características es importante para las predicciones más precisas en este contexto de predicción de beneficios del seguro.

Tarea 4. Ofuscar datos
Lo mejor es ofuscar los datos multiplicando las características numéricas (recuerda que se pueden ver como la matriz 𝑋
) por una matriz invertible 𝑃
.

𝑋′=𝑋×𝑃
Trata de hacerlo y comprueba cómo quedarán los valores de las características después de la transformación. Por cierto, la propiedad de invertibilidad es importante aquí, así que asegúrate de que 𝑃
sea realmente invertible.

Puedes revisar la lección 'Matrices y operaciones matriciales -> Multiplicación de matrices' para recordar la regla de multiplicación de matrices y su implementación con NumPy.

¿Puedes adivinar la edad o los ingresos de los clientes después de la transformación?

Después de la transformación, no es posible adivinar con precisión la edad o los ingresos de los clientes simplemente observando los datos ofuscados resultantes de la multiplicación de la matriz X por la matriz invertible P. Los valores transformados son muy diferentes de los datos originales y no conservan una relación directa con las características originales, lo que dificulta la interpretación y la predicción de las edades o los ingresos de los clientes.

¿Puedes recuperar los datos originales de 𝑋′
si conoces 𝑃
? Intenta comprobarlo a través de los cálculos moviendo 𝑃
del lado derecho de la fórmula anterior al izquierdo. En este caso las reglas de la multiplicación matricial son realmente útiles

Seguramente puedes ver que algunos valores no son exactamente iguales a los de los datos originales. ¿Cuál podría ser la razón de ello?

La razón por la que algunos valores no son exactamente iguales a los datos originales después de la recuperación puede deberse a errores de redondeo y precisión numérica durante el proceso de cálculo. Aunque se utilizó la matriz invertible 𝑃 para ofuscar los datos y luego se intentó recuperar los datos originales multiplicando por la inversa de 𝑃, la precisión numérica limitada en las operaciones matriciales y de punto flotante puede llevar a pequeñas discrepancias entre los valores recuperados y los valores originales.

4 Prueba de que la ofuscación de datos puede funcionar con regresión lineal
En este proyecto la tarea de regresión se ha resuelto con la regresión lineal. Tu siguiente tarea es demostrar analytically que el método de ofuscación no afectará a la regresión lineal en términos de valores predichos, es decir, que sus valores seguirán siendo los mismos. ¿Lo puedes creer? Pues no hace falta que lo creas, ¡tienes que que demostrarlo!

Entonces, los datos están ofuscados y ahora tenemos 𝑋×𝑃
en lugar de tener solo 𝑋
. En consecuencia, hay otros pesos 𝑤𝑃
como
𝑤=(𝑋𝑇𝑋)−1𝑋𝑇𝑦⇒𝑤𝑃=[(𝑋𝑃)𝑇𝑋𝑃]−1(𝑋𝑃)𝑇𝑦
¿Cómo se relacionarían 𝑤
y 𝑤𝑃
si simplificáramos la fórmula de 𝑤𝑃
anterior?

¿Cuáles serían los valores predichos con 𝑤𝑃
?

¿Qué significa esto para la calidad de la regresión lineal si esta se mide mediante la RECM? Revisa el Apéndice B Propiedades de las matrices al final del cuaderno. ¡Allí encontrarás fórmulas muy útiles!

No es necesario escribir código en esta sección, basta con una explicación analítica.

Respuesta

elación entre 𝑤
y 𝑤𝑃
: La fórmula de los pesos de la regresión lineal con los datos ofuscados 𝑤𝑃
se puede expresar como: 𝑤𝑃=[(𝑋𝑃)𝑇𝑋𝑃]−1(𝑋𝑃)𝑇𝑦
. Si simplificamos esta expresión, notaremos que 𝑋𝑇𝑋
se puede representar como 𝑋𝑇𝑋=(𝑋𝑃)𝑇(𝑋𝑃)
, ya que 𝑋𝑃
es simplemente la matriz de datos original 𝑋
multiplicada por la matriz de ofuscación 𝑃
. Por lo tanto, podemos escribir: 𝑤𝑃=[𝑋𝑇𝑋]−1(𝑋𝑃)𝑇𝑦
. Si comparamos esta expresión con la fórmula original de los pesos de la regresión lineal (𝑤
), vemos que son idénticas, lo que significa que los pesos obtenidos después de la ofuscación son los mismos que los de la regresión lineal original. Esto demuestra que la ofuscación de los datos no afecta la estimación de los pesos en la regresión lineal.

Valores predichos con 𝑤𝑃
: Los valores predichos con los datos ofuscados (𝑋×𝑃
) utilizando los pesos 𝑤𝑃
se pueden calcular de la misma manera que con los datos originales. Dado un nuevo conjunto de características 𝑋′, los valores predichos 𝑦̂
se calculan como 𝑋′𝑤𝑃
, donde 𝑤𝑃
es el vector de pesos obtenidos después de la ofuscación.

Implicaciones para la calidad de la regresión lineal: Dado que los pesos de la regresión lineal no se ven afectados por la ofuscación de datos y los valores predichos se calculan de manera similar, la calidad de la regresión lineal medida por el RMSE no se verá afectada por la ofuscación. El RMSE seguirá siendo una medida válida de la precisión del modelo, independientemente de si los datos están ofuscados o no. Esto significa que la capacidad predictiva del modelo de regresión lineal se mantiene incluso después de la ofuscación de datos.

Prueba analítica

Nuestra prueba analítica respalda la conclusión de que la ofuscación de datos puede funcionar con regresión lineal sin afectar la precisión del modelo ni la evaluación de su calidad mediante el RMSE

5 Prueba de regresión lineal con ofuscación de datos
Ahora, probemos que la regresión lineal pueda funcionar, en términos computacionales, con la transformación de ofuscación elegida. Construye un procedimiento o una clase que ejecute la regresión lineal opcionalmente con la ofuscación. Puedes usar una implementación de regresión lineal de scikit-learn o tu propia implementación. Ejecuta la regresión lineal para los datos originales y los ofuscados, compara los valores predichos y los valores de las métricas RMSE y 𝑅2
. ¿Hay alguna diferencia?

Procedimiento

Crea una matriz cuadrada 𝑃
de números aleatorios.- Comprueba que sea invertible. Si no lo es, repite el primer paso hasta obtener una matriz invertible.- <¡ tu comentario aquí !>
Utiliza 𝑋𝑃
como la nueva matriz de características

Observaciones

Los resultados muestran una diferencia muy pequeña en el RMSE entre los datos originales y los ofuscados, con valores prácticamente cero en ambos casos. Además, el coeficiente de determinación 𝑅2
es igual a 1 en ambos conjuntos de datos, lo que sugiere que el modelo se ajusta perfectamente a los datos de entrenamiento y explica toda la variabilidad presente. Sin embargo, es importante tener en cuenta que al obtener un RMSE de 0 y un 𝑅2
de 1, existe la posibilidad de sobreajuste del modelo, especialmente dado que se está prediciendo sobre los mismos datos utilizados para entrenarlo.

Conclusiones
En resumen, el proyecto destaca la relevancia de varias etapas clave en el análisis de datos y el aprendizaje automático. En primer lugar, el preprocesamiento cuidadoso de los datos es fundamental para garantizar la calidad y la coherencia de los conjuntos de datos utilizados. Además, la selección adecuada de modelos de aprendizaje automático y técnicas de ingeniería de características puede marcar una gran diferencia en el rendimiento y la precisión de los resultados obtenidos. También se encontró que la escala de los datos puede influir significativamente en el desempeño de los algoritmos, destacando la importancia de este paso en el proceso. Por último, se demostró que la ofuscación de datos es una medida efectiva para salvaguardar la privacidad y la seguridad de la información sensible, lo que subraya la necesidad de considerar la protección de datos en proyectos de análisis de datos.

Lista de control
Escribe 'x' para verificar. Luego presiona Shift+Enter.

Jupyter Notebook está abierto
El código no tiene errores- [ ] Las celdas están ordenadas de acuerdo con la lógica y el orden de ejecución
Se ha realizado la tarea 1
Está presente el procedimiento que puede devolver k clientes similares para un cliente determinado
Se probó el procedimiento para las cuatro combinaciones propuestas - [ ] Se respondieron las preguntas sobre la escala/distancia- [ ] Se ha realizado la tarea 2
Se construyó y probó el modelo de clasificación aleatoria para todos los niveles de probabilidad - [ ] Se construyó y probó el modelo de clasificación kNN tanto para los datos originales como para los escalados. Se calculó la métrica F1.- [ ] Se ha realizado la tarea 3
Se implementó la solución de regresión lineal mediante operaciones matriciales - [ ] Se calculó la RECM para la solución implementada- [ ] Se ha realizado la tarea 4
Se ofuscaron los datos mediante una matriz aleatoria e invertible P - [ ] Se recuperaron los datos ofuscados y se han mostrado algunos ejemplos - [ ] Se proporcionó la prueba analítica de que la transformación no afecta a la RECM - [ ] Se proporcionó la prueba computacional de que la transformación no afecta a la RECM- [ ] Se han sacado conclusiones
Apéndices
6 Apéndice A: Escribir fórmulas en los cuadernos de Jupyter
Puedes escribir fórmulas en tu Jupyter Notebook utilizando un lenguaje de marcado proporcionado por un sistema de publicación de alta calidad llamado 𝐿𝐴𝑇𝐸𝑋
(se pronuncia como "Lah-tech"). Las fórmulas se verán como las de los libros de texto.

Para incorporar una fórmula a un texto, pon el signo de dólar ($) antes y después del texto de la fórmula, por ejemplo: 12×32=34
or 𝑦=𝑥2,𝑥≥1
.

Si una fórmula debe estar en el mismo párrafo, pon el doble signo de dólar ($$) antes y después del texto de la fórmula, por ejemplo:
𝑥¯=1𝑛∑𝑖=1𝑛𝑥𝑖.
El lenguaje de marcado de LaTeX es muy popular entre las personas que utilizan fórmulas en sus artículos, libros y textos. Puede resultar complicado, pero sus fundamentos son sencillos. Consulta esta ficha de ayuda (materiales en inglés) de dos páginas para aprender a componer las fórmulas más comunes.

7 Apéndice B: Propiedades de las matrices
Las matrices tienen muchas propiedades en cuanto al álgebra lineal. Aquí se enumeran algunas de ellas que pueden ayudarte a la hora de realizar la prueba analítica de este proyecto.

Distributividad 𝐴(𝐵+𝐶)=𝐴𝐵+𝐴𝐶
No conmutatividad 𝐴𝐵≠𝐵𝐴
Propiedad asociativa de la multiplicación (𝐴𝐵)𝐶=𝐴(𝐵𝐶)
Propiedad de identidad multiplicativa 𝐼𝐴=𝐴𝐼=𝐴
𝐴−1𝐴=𝐴𝐴−1=𝐼
(𝐴𝐵)−1=𝐵−1𝐴−1
Reversibilidad de la transposición de un producto de matrices, (𝐴𝐵)𝑇=𝐵𝑇𝐴𝑇
