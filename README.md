
# Clasificación de Ingresos con Spark ML

Este proyecto implementa un modelo de clasificación binaria para predecir si una persona gana más de 50K al año, utilizando PySpark ML y scikit-learn.

## Requisitos y configuración

1. **Python 3** y **pip**
2. **Java (OpenJDK 11+)** para PySpark
3. **Entorno virtual recomendado**

### Configuración rápida (Linux/zsh)

```zsh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt-get install openjdk-11-jdk
```

## Estructura principal

- `Data/adult_income_sample.csv`: Datos de entrada
- `src/notebooks/eda.ipynb`: Notebook principal con todo el flujo de trabajo
- `requirements.txt`: Dependencias
- `sparkMlClass.py`: Ejemplo de pipeline con Spark ML

## Ejecución

### Correr el proyecto

```bash
py src/sparkMlClass.py
```

### Correr el notebook principal

Abre el notebook `src/notebooks/eda.ipynb` y sigue los pasos documentados para:

- Preparar los datos
- Ingeniería de características
- Entrenar y validar modelos
- Probar el modelo con nuevos datos

## Puntos clave

- El flujo completo está documentado en el notebook, paso a paso.
- Incluye comparación de modelos (Logistic Regression, SVM, KNN, LightGBM).
- Se implementa validación cruzada y bootstrapping.
- Ejemplo de prueba con filas manuales y comparación de resultados.

---
*Para detalles y experimentos, consulta el notebook principal.*
