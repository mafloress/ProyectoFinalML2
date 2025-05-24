# Estimación de Pérdidas en Préstamos Incobrables de Lending Club
Miguel Ángel Flores Saldívar

## Uso de LLMs y Agentes IA
Se utilizaron los siguientes LLMs para poder tomar ideas para tomar la base del proyecto y buscar fuentes reales de datos para el proyecto, se hizo la comparativa entre cada respuesta de cada uno y se escogío la mas ad hoc:
- Gemini
- Copilot
- Deepseek
- ChatGPT

Se utilizó adicional para documentación e implementación en Github:
- Jules Google

## Objetivo del Proyecto
El objetivo principal de este proyecto es analizar los datos de préstamos de Lending Club para construir un modelo de aprendizaje automático capaz de predecir la probabilidad de que un préstamo entre en incumplimiento (default). Esto permite estimar las pérdidas potenciales en préstamos que se consideran incobrables y ayuda en la toma de decisiones crediticias.

## Descripción del Conjunto de Datos
Se utiliza el conjunto de datos de Lending Club disponible en Kaggle. Este dataset contiene información detallada sobre préstamos aprobados y rechazados originados entre 2007 y 2018. Para este proyecto, nos enfocamos principalmente en los préstamos aceptados (`accepted_2007_to_2018Q4.csv.gz`) para predecir el incumplimiento de los mismos. El dataset incluye una amplia gama de variables, como el monto del préstamo, la tasa de interés, información del prestatario (ingresos, historial crediticio, empleo), y el estado final del préstamo.

La descarga inicial de los datos se realiza mediante la librería `kagglehub`, utilizando el dataset `wordsforthewise/lending-club`.

## Arquitectura de Pipelines
El proyecto está estructurado en un enfoque de tres pipelines principales para organizar el flujo de trabajo desde los datos crudos hasta las predicciones:

1.  **`Pipeline de Características (Feature Pipeline)`**:
    *   **Entrada**: Datos crudos de Lending Club (ej. `accepted_2007_to_2018Q4.csv.gz`).
    *   **Proceso**: Este pipeline se encarga de la limpieza de datos, transformación de variables (manejo de fechas, conversión de tipos, etc.), imputación de valores faltantes, codificación de variables categóricas y escalado de características numéricas. También incluye la selección de las características más relevantes.
    *   **Salida**: Un conjunto de datos procesado y listo para el entrenamiento del modelo (ej. `processed_lending_club_data.csv`).
    *   **Implementación**: `notebooks/feature_engineering_notebook.ipynb`.

2.  **`Pipeline de Entrenamiento (Training Pipeline)`**:
    *   **Entrada**: Conjunto de datos procesado del pipeline de características.
    *   **Proceso**: Divide los datos en conjuntos de entrenamiento y prueba. Compara diferentes algoritmos de clasificación (usando `LazyPredict`), selecciona un modelo final, lo entrena, y evalúa su rendimiento. Utiliza MLflow para el seguimiento de experimentos, parámetros, métricas y el versionado del modelo.
    *   **Salida**: Un modelo entrenado (ej. `trained_model.pkl`) y artefactos registrados en MLflow.
    *   **Implementación**: `notebooks/model_training_notebook.ipynb`.

3.  **`Pipeline de Inferencia Batch (Batch Inference Pipeline)`**:
    *   **Entrada**: Nuevos datos (con las mismas características procesadas que en el entrenamiento) y el modelo entrenado.
    *   **Proceso**: Carga el modelo entrenado y lo utiliza para generar predicciones (clase y probabilidad de default) sobre un nuevo lote de datos.
    *   **Salida**: Un archivo con las predicciones para el lote de datos (ej. `batch_predictions.csv`).
    *   **Implementación**: `scripts/batch_inference_script.py` (y conceptualmente `notebooks/batch_inference_notebook.ipynb`).

Adicionalmente, se incluye un notebook de Análisis Exploratorio de Datos (`notebooks/eda_notebook.ipynb`) para una comprensión inicial del dataset.

## Decisiones de Modelado

*   **`Variable Objetivo`**:
    *   Se define una variable binaria `is_default`. Se considera como `1` (default) a los préstamos con `loan_status` en: 'Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', y 'Late (31-120 days)'. El resto de los estados se consideran `0` (no default).

*   **`Ingeniería de Características`**:
    *   Manejo de fechas: Extracción de mes y año de variables como `issue_d`, `earliest_cr_line`. Creación de `credit_history_length`.
    *   Conversión de `emp_length` (antigüedad laboral) a valores numéricos.
    *   Codificación de variables categóricas: Se utiliza `pd.get_dummies` (one-hot encoding) para convertir variables categóricas en numéricas, eliminando la primera categoría para evitar multicolinealidad (`drop_first=True`).
    *   Escalado de características numéricas: Se aplica `StandardScaler` de scikit-learn para normalizar las características numéricas.
    *   Limpieza: Eliminación de columnas identificadoras, de texto libre, con alto porcentaje de valores faltantes o baja varianza.
    *   Imputación: Mediana para variables numéricas y moda para variables categóricas.

*   **`Selección de Características`**:
    *   Se utiliza `mutual_info_classif` de scikit-learn para evaluar la importancia de cada característica con respecto a la variable objetivo.
    *   Se seleccionan las **top 75 características** con mayor puntuación de información mutua para entrenar el modelo final.

*   **`Comparación de Modelos`**:
    *   Se emplea `LazyClassifier` de la librería `LazyPredict` para obtener una evaluación rápida del rendimiento de múltiples algoritmos de clasificación sobre el dataset y guiar la selección del modelo.

*   **`Modelo Seleccionado`**:
    *   Se selecciona `LGBMClassifier` (LightGBM) como el modelo final para el entrenamiento. Esta elección se basa en su robustez, eficiencia con datasets grandes, buen manejo de características categóricas y rendimiento general en problemas de clasificación similares, especialmente con datos desbalanceados.

*   **`Seguimiento con MLflow`**:
    *   Se utiliza MLflow para el seguimiento de los experimentos de entrenamiento.
    *   Se registran:
        *   Parámetros del modelo.
        *   Métricas de evaluación (Accuracy, Precision, Recall, F1-score para la clase 'default', AUC).
        *   Artefactos, como la matriz de confusión.
        *   El modelo entrenado.

## Resultados Principales y Métricas
El modelo `LGBMClassifier` fue entrenado y evaluado sobre el conjunto de prueba. Las métricas clave obtenidas (enfocadas en la predicción de la clase 'default') se registran en MLflow. A modo de ejemplo, un rendimiento típico bueno para este tipo de problema (considerando el desbalance de clases) podría ser:

*   **AUC (Area Under the ROC Curve)**: ~0.70 - 0.75. Un valor de 0.5 indica un modelo aleatorio, mientras que 1.0 es perfecto.
*   **F1-score (clase default)**: ~0.30 - 0.50. Es una media armónica entre Precision y Recall, útil para clases desbalanceadas.
*   **Precision (clase default)**: ~0.40 - 0.60. De los préstamos predichos como default, qué porcentaje realmente lo fue.
*   **Recall (clase default)**: ~0.25 - 0.45. De todos los préstamos que realmente fueron default, qué porcentaje se identificó correctamente.

*Nota: Estos valores son ilustrativos. Los valores exactos se encuentran en la ejecución específica del notebook `model_training_notebook.ipynb` y en el UI de MLflow.*

La matriz de confusión, también registrada en MLflow, proporciona una visión detallada del número de Verdaderos Positivos, Falsos Positivos, Verdaderos Negativos y Falsos Negativos.

## Cómo Ejecutar el Proyecto

1.  **Clonar el repositorio**:
    ```bash
    git clone https://placeholder.url/lending-club-loan-default.git
    cd lending-club-loan-default
    ```

2.  **Crear y activar un entorno virtual** (recomendado):
    ```bash
    python -m venv venv
    ```
    En Linux/macOS:
    ```bash
    source venv/bin/activate
    ```
    En Windows:
    ```bash
    venv\\Scripts\\activate
    ```

3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar los notebooks en orden**:
    Asegúrese de que todos los archivos estén organizados según la "Estructura del Proyecto" descrita a continuación antes de ejecutar los scripts o notebooks.

    Se recomienda ejecutar los scripts/notebooks en la siguiente secuencia:

    *   **`scripts/download_data.py` (Script de Python, ejecutar primero si los datos no están disponibles)**:
        ```bash
        python scripts/download_data.py 
        ```
        Este script utiliza `kagglehub` para descargar el dataset. Asegúrese de tener configuradas sus credenciales de Kaggle (`kaggle.json` en `~/.kaggle/` o las variables de entorno `KAGGLE_USERNAME` y `KAGGLE_KEY`). Los datos descargados por `kagglehub` se almacenan en `~/.cache/kagglehub/`. El notebook de ingeniería de características espera encontrar el archivo `accepted_2007_to_2018Q4.csv.gz` en esa ubicación.

    *   **1. `notebooks/eda_notebook.ipynb`**:
        *   Propósito: Análisis exploratorio de datos (opcional, pero recomendado para entender los datos).
        *   No genera salidas directas para los siguientes notebooks, pero informa las decisiones de ingeniería de características.

    *   **2. `notebooks/feature_engineering_notebook.ipynb`**:
        *   Propósito: Realiza la limpieza de datos, transformación e ingeniería de características.
        *   Salida principal: `data/processed/processed_lending_club_data.csv`.

    *   **3. `notebooks/model_training_notebook.ipynb`**:
        *   Propósito: Entrena el modelo `LGBMClassifier`, realiza el seguimiento con MLflow y guarda el modelo.
        *   Entrada: `data/processed/processed_lending_club_data.csv`.
        *   Salidas principales: `models/trained_model.pkl`, experimentos y artefactos en MLflow.

    *   **4. `scripts/batch_inference_script.py` (o `notebooks/batch_inference_notebook.ipynb` para experimentación)**:
        *   Propósito: Demuestra cómo cargar el modelo entrenado y realizar predicciones en un lote de datos.
        *   Entradas: `models/trained_model.pkl`, `data/processed/processed_lending_club_data.csv` (para simular un batch).
        *   Salida principal: `data/predictions/batch_predictions.csv`.
    
    *   **5. `app/chainlit_app.py` (Aplicación interactiva)**:
        ```bash
        chainlit run app/chainlit_app.py -w
        ```
        Asegúrese de que `models/trained_model.pkl` y `data/processed/processed_lending_club_data.csv` existan en las rutas correctas relativas al proyecto.

5.  **Visualizar experimentos con MLflow**:
    Desde la terminal, en el directorio raíz del proyecto, ejecute:
    ```bash
    mlflow ui
    ```
    Esto iniciará un servidor local (generalmente en `http://127.0.0.1:5000`) donde podrá ver los experimentos, corridas, parámetros, métricas y artefactos registrados.

## Estructura del Proyecto
```
lending_club_project/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       ├── training_pipeline.yml
│       └── batch_inference.yml
├── data/
│   ├── processed/
│   │   └── processed_lending_club_data.csv
│   └── predictions/
│       └── batch_predictions.csv
├── notebooks/
│   ├── eda_notebook.ipynb
│   ├── feature_engineering_notebook.ipynb
│   ├── model_training_notebook.ipynb
│   └── batch_inference_notebook.ipynb
├── scripts/
│   ├── download_data.py
│   └── batch_inference_script.py
├── app/
│   └── chainlit_app.py
├── models/
│   └── trained_model.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```

**Nota**: El script `scripts/download_data.py` se añadió para facilitar la obtención del dataset. El directorio `data/` es donde se almacenan los datos procesados y las predicciones. El directorio `mlruns/` se crea automáticamente por MLflow en el directorio raíz la primera vez que se ejecuta un script con logging de MLflow (si se ejecuta desde el raíz).
**Nota**: El script `download_data.py` se añadió para facilitar la obtención del dataset, y el directorio `data/` es donde `kagglehub` podría almacenar los datos o donde el usuario podría moverlos. El directorio `mlruns/` se crea automáticamente por MLflow en el directorio raíz la primera vez que se ejecuta un script con logging de MLflow.
