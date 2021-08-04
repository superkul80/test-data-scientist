# Ejercicio práctico para Data Scientist en Deacero.

Debe realizar un fork de este repositorio para desarrollar y entregar su trabajo.

1. Obtén los datos de las siguientes fuentes desde las apis:

| Dataset                   | Url                                                               |
| ------------------------- | ----------------------------------------------------------------- |
| Hotel                     | https://analytics.deacero.com/api/expuesta/sqlexp/api_key         |


Nota: El api_key válido se proporciona por correo. En caso de estar interesado en aplicar al test puede enviar correo a <yacosta@deacero.com>

Supongamos que fuimos contratados como consultores para responder esta pregunta clave: ¿cuándo un cliente cancelará o no llegará a su reservación?

Esto puede ayudar a un hotel a planificar cosas como las necesidades personales y alimentarias. Quizás algunos hoteles también usan este modelo para ofrecer más habitaciones de las que tienen para ganar más dinero, o cosas por el estilo.

El conjunto de datos del archivo Hotel.csv contiene información de reserva para un hotel urbano y un hotel resort, e incluye información como cuándo se realizó la reserva, duración de la estadía, la cantidad de adultos, niños y / o bebés, y la cantidad de espacios de estacionamiento disponibles, entre otras cosas. Toda la información de identificación personal se ha eliminado de los datos. Ambos hoteles están ubicados en Portugal.

Preguntas para responder: 

●	¿De dónde vienen los huéspedes? 

●	¿Cuánto pagan los huéspedes por una habitación por noche en promedio? 

●	¿Cómo varía el precio por noche durante el año? 

●	¿Cuáles son los meses más ocupados?  

●	¿Cuánto tiempo se queda la gente en los hoteles (noches)?
 
●	Reservas por segmento de mercado 

●	¿Cuántas reservas se cancelaron?  

●	¿Qué mes tiene el mayor número de cancelaciones?  

Antes de intentar hacer predicciones asegúrense de hacer una correcta limpieza de los datos.

●	Hacer tabla de correlación para las variables 

●	Encontrar las mejores variables para predecir cancelaciones 

●	Predecir a través de 3 diferentes modelos. 

●	Para cada modelo elegido, utilice un K-Fold de 4 y evalúe el cross_val_score 

●	¿Cuál es el modelo con mayor performance? 



Instrucciones: 

1.	Responde las preguntas planteadas en la descripción del problema, utilizando R, Python o ambas.
2.	Seguir todos los pasos para la realización del modelo predictivo y análisis correspondiente. 
3.	Predecir si reservación será cancelada en base de la información de cancelaciones pasadas. La columna en cuestión es "is_canceled" (1 es sí, 0 es no). 
4.	Hacer una presentación ejecutiva de 5 slides en los que presenten sus resultados a una audiencia de negocio. Esta audiencia no sabe nada de algoritmos, ni de modelos, solamente quieren resolver el problema de cancelaciones. Pueden usar PPT o subir un PDF con sus láminas.


Los ejercicios deben realizarse en Python, además de un documento donde se contesten las preguntas y se muestren los resultados de las transformaciones. En caso de trabajar con notebooks de jupyter puede exportarse en HTML. 

Una vez concluido el reto se debe comunicar al correo correspondiente con la liga al repositorio de github final para evaluar las respuestas.


Suerte a todos!!! :metal: :nerd_face: :computer:
