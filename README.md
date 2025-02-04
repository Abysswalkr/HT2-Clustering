# HT2 - Clustering

## Universidad del Valle de Guatemala  
**Facultad de Ingeniería**  
**Ciencias de la Computación y Tecnologías de la Información**  

## Descripción del Proyecto
Este repositorio contiene la **Hoja de Trabajo 2 - Clustering** para la materia de **Minería de Datos**. El objetivo del proyecto es realizar **segmentación de especies de Iris** utilizando técnicas de **Análisis de Clusters**, en particular **K-Means Clustering** y el **método del codo**.

## Objetivos
1. **Explorar y visualizar los datos** del dataset Iris.
2. **Aplicar K-Means Clustering** para identificar posibles grupos dentro de los datos.
3. **Estandarizar los datos** para mejorar la precisión del clustering.
4. **Determinar el número óptimo de clusters** utilizando el **método del codo** y la librería `kneed`.
5. **Comparar los resultados obtenidos** con los datos reales de especies de Iris.
6. **Evaluar la precisión del clustering** utilizando el **Índice Rand Ajustado (ARI)**.

## Estructura del Repositorio
```
HT2-Clustering/
│── seccion_1/
│   ├── ejercicio_1.py
│   ├── ejercicio_2.py
│   ├── ejercicio_3.py
│   ├── ejercicio_4.py
│   ├── ejercicio_5.py
│   ├── ejercicio_6.py
│
│── seccion_2/
│   ├── ejercicio_1.py
│   ├── ejercicio_2.py
│   ├── ejercicio_3.py
│   ├── ejercicio_4.py
│   ├── ejercicio_5.py
│   ├── ejercicio_6.py
│
│── seccion_3/
│   ├── ejercicio_1.py  # Implementación con kneed
│
│── iris.csv  # Dataset de Iris sin etiquetas
│── iris-con-respuestas.csv  # Dataset con especies reales
│── README.md  # Documentación del proyecto
```

## Descripción de las Secciones
### **Sección 1: Análisis con la Forma del Sépalo**
- Visualización de los datos para detectar grupos.
- Aplicación de K-Means con `k=2` y comparación de resultados.
- Estandarización de datos y reevaluación de clusters.
- Uso del método del codo para determinar el número óptimo de clusters.
- Comparación con datos reales.

### **Sección 2: Análisis con la Forma del Pétalo**
- Repetición del proceso usando características del pétalo.
- Comparación de la efectividad de la segmentación en comparación con la forma del sépalo.

### **Sección 3: Evaluación con `kneed` y ARI**
- Implementación del método del codo usando la librería `kneed`.
- Comparación entre el método del codo manual y `kneed`.
- Evaluación de diferencias y análisis de precisión respecto a los datos reales.
- Determinación del número correcto de clusters y análisis de resultados.

## Conclusiones
- **El método del codo no siempre coincide con la segmentación biológica real.** En algunos casos, `kneed` puede sugerir **k=2**, pero el análisis con ARI confirma que **k=3** es más preciso.
- **Es importante complementar el método del codo con métricas adicionales**, como el Índice Rand Ajustado y el Coeficiente de Silueta.
- **La forma del pétalo ofrece una mejor separación de clusters** que la forma del sépalo.
- **El uso de `kneed` mejora la detección del número óptimo de clusters**, pero siempre es recomendable validar los resultados con métricas de calidad de clustering.

## Requerimientos
Para ejecutar el código en este repositorio, asegúrate de tener instaladas las siguientes librerías en Python:
```bash
pip install numpy pandas matplotlib scikit-learn kneed
```

## Autores
- **Angel Andres Herrarte Lorenzana (22873)**
- **José Luis Gramajo Moraga (22907)**

## Repositorio de GitHub
[HT2-Clustering Repository](https://github.com/Abysswalkr/HT2-Clustering.git)

