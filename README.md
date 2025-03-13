# Proyecto Integrador: Crime Detection

## Descripción
En el siguiente ReadMe se detallan los detalles del proyecto de crime detection. El proyecto se realizo como Proyecto Integrador para la Universidad San Francisco de Quito.

## Estructura del Proyecto
```
├── data               
│   ├── raw
│   ├── clean
├── docs               
├── models             # Carpeta para modelos
│   ├── i3d.py         # Modelo Inception I3D
├── notebooks          
│   ├── 1_exploratory_data_analysis.ipynb
│   ├── 2_data_wrangling.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_models.ipynb
├── scripts           
│   ├── __init__.py
│   ├── blackAndWhite.py 
│   ├── createTable.py
│   ├── duplicados.py 
├── environment.yml 
```

## Instalación
Para instalar las dependencias necesarias, ejecute:
```bash
conda env create -f environment.yml
```

## Uso
El siguiente proyecto utiliza el modelo i3d para entrenar un modelo de reconocimiento de crimenes. Se pueden apreciar los resultados del proyecto en 4_models.ipynb

## Contribución


## Contacto
Si se necesita mayor información, escribir al mail andres.herrerag4@gmail.com

## Referencias

[1] J. Carreira and A. Zisserman, “Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset,” 
    arXiv preprint arXiv:1705.07750, 2017. [Online]. Available: https://arxiv.org/pdf/1705.07750v1.pdf  

[2] brunomcebola, "Modificación del modelo I3D en el repositorio original,"  
    GitHub, [30/08/2024]. [Online]. Available: https://github.com/google-deepmind/kinetics-i3d/pull/127