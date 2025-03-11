# Proyecto Integrador: Crime Detection

## Descripción
En el siguiente ReadMe se detallan los detalles del proyecto de crime detection. El proyecto se realizo como Proyecto Integrador para la Universidad San Francisco de Quito.

## Estructura del Proyecto
```
├── data               # Datos del proyecto
│   ├── raw
│   ├── clean
├── docs               # Documentación del proyecto
├── notebooks          # Notebooks de Jupyter
│   ├── 1_exploratory_data_analysis.ipynb
├── scripts            # Scripts de procesamiento
│   ├── __init__.py
│   ├── blackAndWhite.py 
│   ├── createTable.py
│   ├── duplicados.py 
├── environment.yml   # Dependencias del proyecto
```

## Instalación
Para instalar las dependencias necesarias, ejecute:
```bash
conda env create -f environment.yml
```

## Uso
Instrucciones sobre cómo ejecutar el proyecto, incluyendo ejemplos de comandos.

## Contribución
Guía para contribuir al proyecto, incluyendo cómo clonar el repositorio y enviar pull requests.

## Contacto
Si se necesita mayor información, escribir al mail andres.herrerag4@gmail.com

## References

- Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. *arXiv preprint arXiv:1705.07750*.  
  Available at: [https://arxiv.org/abs/1705.07750](https://arxiv.org/abs/1705.07750)

- The I3D models used in this project are based on the implementation and pre-trained checkpoints provided by DeepMind in their [Kinetics-I3D repository](https://github.com/deepmind/kinetics-i3d).