# Clasificación de Ingresos con Spark ML

Este proyecto implementa un modelo de clasificación binaria con PySpark ML para predecir si una persona gana más de 50K al año, utilizando datos demográficos y laborales.

## Configuración del entorno (Linux/zsh)

### Crear un entorno virtual con venv

```zsh
python3 -m venv venv
source venv/bin/activate
```

### Instalar dependencias**

```zsh
pip install -r requirements.txt
```

### Instalar Java (requerido para PySpark)**

```zsh
sudo apt-get install openjdk-11-jdk
java -version
```

## Estructura del proyecto

- `src/` : Código fuente principal (incluye `sparkMlClass.py`)
- `main.py` : Script de entrada para ejecutar el pipeline
- `requirements.txt` : Dependencias
- `Data/` : Datos de entrada
- `tests/` : Pruebas unitarias (opcional)

## Ejecución

1. Ejecuta el pipeline:

```zsh
python main.py
```
