# SmartBazar - Dashboard Inteligente

Este es un dashboard interactivo construido con Streamlit para el proyecto "SmartBazar". El dashboard incluye análisis exploratorio de datos, reglas de asociación, modelos predictivos y pronóstico de ingresos.

## Requisitos Previos

Asegúrate de tener Python instalado en tu sistema (preferiblemente Python 3.8 o superior).

## Instrucciones para levantar el proyecto

Sigue los siguientes pasos para ejecutar el dashboard en tu máquina local:

### 1. Clonar el repositorio (si aplica) o abrir la carpeta del proyecto
Abre una terminal o consola de comandos (Command Prompt, PowerShell o la terminal de tu IDE como VS Code) y navega hasta la carpeta del proyecto `smart-bazar`.

### 2. (Recomendado) Crear y activar un entorno virtual
Es una buena práctica usar un entorno virtual para no tener conflictos de dependencias.

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En macOS y Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar las dependencias
Una vez dentro del entorno virtual, instala las librerías necesarias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
```

Esto instalará librerías como `streamlit`, `pandas`, `scikit-learn`, entre otras.

### 4. Ejecutar la aplicación de Streamlit
Para levantar el dashboard, ejecuta el archivo principal `app.py` utilizando Streamlit:

```bash
streamlit run app.py
```

### 5. Ver el Dashboard
Después de ejecutar el comando anterior, Streamlit te mostrará en la consola unas URLs (por ejemplo, `http://localhost:8501`). Normalmente, tu navegador web predeterminado se abrirá automáticamente mostrando el dashboard. Si no se abre automáticamente, simplemente copia y pega la URL Local en tu navegador.

## Paneles Disponibles
El dashboard cuenta con las siguientes secciones en la barra lateral:
- **Panel 1A: EDA y Clustering**
- **Panel 1B: Reglas de Asociación**
- **Panel 2: Predicción de Pagos**
- **Panel 3: Pronóstico de Ingresos**
- **Panel 4: Punto de Venta (CRUD)**
