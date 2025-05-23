import chainlit as cl
import pickle
import pandas as pd
import os

# --- Configuración de Rutas (ajustar si es necesario) ---
# Asumimos que los archivos están en el mismo directorio que la app,
# o en /app/ si se ejecuta en el Docker del paso anterior.
MODEL_PATH = os.getenv("MODEL_PATH", 'trained_model.pkl')
DATA_PATH = os.getenv("DATA_PATH", 'processed_lending_club_data.csv')

# --- Carga Global de Modelo y Datos ---
# Estos se cargarán una vez cuando la aplicación Chainlit se inicie.
model = None
processed_data_sample = None
load_error = None

try:
    print(f"Intentando cargar el modelo desde: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Modelo cargado exitosamente.")

    print(f"Intentando cargar los datos procesados desde: {DATA_PATH}")
    df_full_processed = pd.read_csv(DATA_PATH)
    print(f"Datos procesados cargados. Shape: {df_full_processed.shape}")
    
    # Preparar un conjunto de datos de muestra (sin la columna 'is_default')
    if 'is_default' in df_full_processed.columns:
        processed_data_features = df_full_processed.drop(columns=['is_default'])
    else:
        processed_data_features = df_full_processed.copy()
    
    # Tomar una muestra para usar en la predicción (ej. las primeras 5 filas para elegir)
    # El callback tomará la primera de estas.
    processed_data_sample = processed_data_features.head() 
    print("Muestra de datos preparada.")

except FileNotFoundError as e:
    error_message = f"Error de archivo no encontrado: {e}. Asegúrese de que '{MODEL_PATH}' y '{DATA_PATH}' estén disponibles."
    print(error_message)
    load_error = error_message
except Exception as e:
    error_message = f"Ocurrió un error general al cargar modelo/datos: {e}"
    print(error_message)
    load_error = error_message

@cl.on_chat_start
async def start():
    """
    Función que se ejecuta al inicio de una nueva sesión de chat.
    """
    if load_error:
        await cl.Message(
            content=f"Error al iniciar la aplicación: {load_error}\nPor favor, verifique los archivos del modelo y de datos."
        ).send()
        return

    if model is None or processed_data_sample is None:
        await cl.Message(
            content="Error: El modelo o los datos de ejemplo no pudieron ser cargados. Revise los logs del servidor."
        ).send()
        return

    await cl.Message(
        content="Bienvenido al Estimador de Incumplimiento de Préstamos de Lending Club."
    ).send()

    actions = [
        cl.Action(
            name="predict_sample", 
            label="Predecir para un caso de ejemplo", 
            description="Utiliza un caso de ejemplo de los datos procesados para generar una predicción."
        )
    ]
    await cl.Message(content="Haga clic en el botón para obtener una predicción:", actions=actions).send()

@cl.action_callback("predict_sample")
async def on_action_predict_sample(action: cl.Action):
    """
    Se ejecuta cuando el usuario hace clic en el botón 'predict_sample'.
    """
    await action.remove() # Opcional: eliminar el botón después de hacer clic

    if model is None or processed_data_sample is None or processed_data_sample.empty:
        await cl.Message(
            content="Lo sentimos, no podemos procesar la predicción en este momento debido a un problema con el modelo o los datos."
        ).send()
        return

    try:
        # Seleccionar la primera fila de la muestra de datos preparada
        sample_features_df = processed_data_sample.iloc[[0]]
        
        # Verificar que las columnas coincidan con las esperadas por el modelo
        # Esto es crucial si el modelo es sensible al orden o número de características
        if hasattr(model, 'feature_name_') and list(sample_features_df.columns) != list(model.feature_name_):
            print("Alineando columnas del sample con las del modelo...")
            sample_features_df = sample_features_df[model.feature_name_]
        elif hasattr(model, 'n_features_in_') and sample_features_df.shape[1] != model.n_features_in_:
             error_msg = f"Error: El modelo espera {model.n_features_in_} características, pero la muestra tiene {sample_features_df.shape[1]}."
             print(error_msg)
             await cl.Message(content=error_msg).send()
             return


        # Realizar predicciones
        prediction = model.predict(sample_features_df)[0]
        probability = model.predict_proba(sample_features_df)[0] # Devuelve prob para ambas clases

        # La probabilidad de la clase '1' (default)
        probability_default = probability[1] 

        # Mostrar características de ejemplo (opcional, primeras 5 para brevedad)
        sample_display = sample_features_df.iloc[0, :5].to_dict()
        sample_info_str = ", ".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in sample_display.items()])
        
        response_content = (
            f"Se utilizó un caso de ejemplo para la predicción.\n"
            f"Primeras 5 características del caso: {sample_info_str}...\n\n"
            f"**Predicción de Incumplimiento:**\n"
            f"- Probabilidad de Incumplimiento: **{probability_default*100:.2f}%**\n"
            f"- Clase Predicha: **{'Incumplimiento' if prediction == 1 else 'No Incumplimiento'}**"
        )
        
        await cl.Message(content=response_content).send()

    except Exception as e:
        error_msg = f"Ocurrió un error al generar la predicción: {e}"
        print(error_msg)
        await cl.Message(content=error_msg).send()

# Para ejecutar esta aplicación Chainlit:
# 1. Guarde este archivo como `chainlit_app.py`.
# 2. Asegúrese de que `trained_model.pkl` y `processed_lending_club_data.csv` estén en el mismo directorio.
# 3. Ejecute desde la terminal: `chainlit run chainlit_app.py -w`
#    (la opción -w habilita el auto-reload para desarrollo)
