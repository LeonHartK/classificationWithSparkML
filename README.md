# Clasificación de Ingresos con Spark ML

Este proyecto implementa un modelo de clasificación binaria con PySpark ML para predecir si una persona gana más de 50K al año, utilizando datos demográficos y laborales.

**Crear un entorno virtual con venv**

```powershell
python -m venv venv
```

**Activar el entorno virtual**

```powershell
.\venv\Scripts\Activate.ps1
```

Si aparece error de ejecución de scripts, habilita permisos con:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

y vuelve a activar:

```powershell
.\venv\Scripts\Activate.ps1
```

**Instalar dependencias**

```powershell
pip install -r requirements.txt
```

## Ejecución

1. Corre el script de clasificación:

```powershell
python sparkMlClass.py
```