# Proyecto Integrador: Crime Detection

## Descripción

Este proyecto tiene como objetivo el desarrollo de modelos de aprendizaje profundo capaces de detectar crímenes en videos provenientes de cámaras de vigilancia. Fue desarrollado como parte del Proyecto Integrador de la Universidad San Francisco de Quito.

Se utilizaron modelos de video clasificación como I3D (Inception 3D) y una arquitectura combinada con ConvLSTM2D para detectar actividades criminales. El sistema está en fase de prototipo y puede ser visualizado en acción en el archivo `Proyecto/app/demo.ipynb`.

## Estructura del Proyecto

```
├── data                  # Datos utilizados en el entrenamiento y prueba
│   ├── raw               # Datos originales
│   ├── ml                # Datos formateados para modelos
│   ├── clean             # Datos limpios
├── models                # Modelos implementados
│   ├── i3d.py            # Implementación del modelo I3D
│   ├── KerasI3D.py       # Versión Keras del modelo I3D
├── notebooks             # Notebooks del desarrollo
│   ├── 1_exploratory_data_analysis.ipynb
│   ├── 2_data_wrangling.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_models.ipynb    # Entrenamiento y evaluación de modelos
├── scripts               # Scripts auxiliares
│   ├── __init__.py
│   ├── blackAndWhite.py  # Detección de cambios RGB en videos
│   ├── createTable.py    # Etiquetado de videos con PCB, StartFrame, EndFrame
│   ├── duplicados.py     # Verificación de duplicados en los datos
├── environment.yml       # Archivo de entorno Conda
├── requirements.txt      # Requisitos para instalación vía pip
```

## Dataset

Se utilizó el dataset **UCF-Crime**, un conjunto público de videos de vigilancia con 13 clases de crímenes y un total de 1900 videos. Para este proyecto, se seleccionaron las siguientes 10 clases:

* Abuse
* Arson
* Assault
* Burglary
* Fighting
* Robbery
* Shooting
* Shoplifting
* Stealing
* Vandalism

El dataset fue dividido de la siguiente forma:

* **Entrenamiento:** 516 videos
* **Validación:** 65 videos
* **Prueba:** 65 videos

## Modelos

Se implementaron dos modelos principales:

1. **I3D (RGB-Inception3D)**
2. **I3D combinado con ConvLSTM2D**

Estos modelos fueron seleccionados en base a su desempeño en el estado del arte según la literatura revisada.

### Métricas de Evaluación

* Accuracy
* Precision
* F1-Score
* AUC

### Resultados

| Métrica         | I3D  | I3D + ConvLSTM |
| --------------- | ---- | -------------- |
| Accuracy        | 0.63 | 0.68           |
| Precisión (avg) | 0.63 | 0.68           |
| F1-score (avg)  | 0.63 | 0.68           |
| AUC             | 0.64 | 0.73           |

## Instalación

Con **conda**:

```bash
conda env create -f environment.yml
```

O con **pip**:

```bash
pip install -r requirements.txt
```

## Uso

Actualmente no hay una interfaz gráfica o web. El enfoque principal está en el desarrollo y evaluación de modelos. Se pueden ejecutar los notebooks en la carpeta `notebooks/` para seguir el flujo completo del proyecto.

Para ver un ejemplo de cómo podría utilizarse el modelo en tiempo real, consulte:

```
Proyecto/app/demo.ipynb
```

## Scripts

* `blackAndWhite.py`: Detecta cambios significativos en las características de color RGB en los videos.
* `createTable.py`: Etiqueta los videos con datos como PCB, StartFrame y EndFrame.
* `duplicados.py`: Verifica que no existan duplicados en los datos.

Estos scripts pueden ejecutarse de forma independiente y no requieren un orden específico.

## Limitaciones

* El sistema está limitado por la **cantidad de datos disponibles**, lo cual impacta la capacidad de generalización del modelo.
* Futuras mejoras se enfocarán en un **mejor manejo y aumento de los datos** para mejorar el desempeño.

## Contacto

Para mayor información o consultas, por favor escribir a:
[andres.herrerag4@gmail.com](mailto:andres.herrerag4@gmail.com)

## Licencia

Este proyecto es propiedad de la **Universidad San Francisco de Quito**. Su uso o distribución está restringido a fines académicos, salvo autorización expresa.
