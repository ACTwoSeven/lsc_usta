# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:22:17 2025

@author: Andre
"""
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Input, Masking, LSTM, Dropout, Dense
from tensorflow.keras.models import Model

# 1. Función para recrear el modelo y cargar pesos (basado en nuestra solución anterior)
def recrear_modelo_y_cargar_pesos(ruta_modelo, ruta_clases):
    try:
        # Cargar clases para determinar num_classes
        clases = np.load(ruta_clases)
        num_clases = len(clases)
        
        # Recrear la arquitectura del modelo
        input_dim = 63  # Landmarks de mano (21 puntos x 3 coordenadas)
        
        inp = Input(shape=(None, input_dim))
        x = Masking(mask_value=0.0)(inp)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(num_clases, activation='softmax')(x)
        
        modelo = Model(inp, out)
        modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Intentar cargar solo los pesos
        try:
            modelo.load_weights(ruta_modelo)
            print("¡Pesos cargados exitosamente en el modelo recreado!")
        except:
            # Si falló, intentar cargar completo con custom_objects
            try:
                modelo_original = tf.keras.models.load_model(
                    ruta_modelo, 
                    compile=False, 
                    custom_objects={'NotEqual': tf.math.not_equal}
                )
                temp_weights = '/tmp/temp_weights.h5'
                modelo_original.save_weights(temp_weights)
                modelo.load_weights(temp_weights)
                print("¡Pesos extraídos y cargados exitosamente!")
            except Exception as e:
                print(f"Error al extraer pesos: {e}")
                return None
            
        return modelo, clases
    except Exception as e:
        print(f"Error al recrear modelo: {e}")
        return None, None

# 2. Función para extraer landmarks vectorizados
def extract_landmark_vector(landmarks):
    """Dado landmarks (list of 21), devuelve un vector (63,) centrado y normalizado."""
    lm = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    # centrar en muñeca
    lm[:, :2] -= lm[0, :2]
    # escalar para que la distancia máxima en XY sea 1
    scale = np.max(np.linalg.norm(lm[:, :2], axis=1))
    if scale > 0:
        lm[:, :2] /= scale
    return lm.flatten()

# 3. Función principal para reconocimiento en tiempo real
def run_lsc_recognition(model_path, classes_path, buffer_size=4):
    print("Cargando modelo y clases...")
    model, CLASSES = recrear_modelo_y_cargar_pesos(model_path, classes_path)
    
    if model is None:
        print("Error: No se pudo cargar el modelo")
        return
    
    print(f"Modelo cargado. Clases disponibles: {CLASSES}")
    
    # Inicializar buffer para secuencia
    buffer = deque(maxlen=buffer_size)
    
    # Inicializar MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Configurar cámara
    print("Iniciando cámara...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")
    
    # Variables para mostrar la letra actual
    current_letter = ""
    prediction_confidence = 0.0
    confidence_threshold = 0.7  # Umbral para mostrar predicciones confiables
    
    print("Presiona ESC para salir...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame de la cámara")
            break
        
        # Voltear horizontalmente (efecto espejo)
        frame = cv2.flip(frame, 1)
        
        # Procesar frame con MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Dibujar información en el frame
        frame_height, frame_width = frame.shape[:2]
        info_panel = np.zeros((frame_height, 200, 3), dtype=np.uint8)
        
        # Detectar y dibujar landmarks de la mano
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
            )
            
            # Extraer vector de landmarks
            vec = extract_landmark_vector(hand_landmarks.landmark)
            buffer.append(vec)
        else:
            # Si no hay mano, repetir último vector o añadir ceros
            if len(buffer) > 0:
                buffer.append(buffer[-1])
            else:
                buffer.append(np.zeros(63, dtype=np.float32))
        
        # Realizar inferencia cuando el buffer está lleno
        if len(buffer) == buffer_size:
            seq = np.stack(buffer)[np.newaxis, ...]  # (1, buffer_size, 63)
            probs = model.predict(seq, verbose=0)[0]
            idx = np.argmax(probs)
            letter = CLASSES[idx]
            conf = probs[idx]
            
            # Actualizar letra actual solo si la confianza es alta
            if conf > confidence_threshold:
                current_letter = letter
                prediction_confidence = conf
        
        # Dibujar información en el panel lateral
        cv2.putText(info_panel, "LSC Dinámico", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(info_panel, "Letra:", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
        # Mostrar letra con confianza
        if current_letter:
            cv2.putText(info_panel, current_letter.upper(), (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, 
                        (0, 255, 0) if prediction_confidence > 0.8 else (0, 200, 200), 
                        3)
            
            cv2.putText(info_panel, f"Conf: {prediction_confidence:.2f}", (10, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(info_panel, "---", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 200), 3)
        
        # Mostrar estado del buffer
        cv2.putText(info_panel, f"Buffer: {len(buffer)}/{buffer_size}", (10, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Combinar frame y panel de información
        combined_frame = np.hstack((frame, info_panel))
        
        # Mostrar el frame combinado
        cv2.imshow('Reconocimiento LSC', combined_frame)
        
        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Aplicación finalizada")

# Ejemplo de uso
if __name__ == "__main__":
    MODEL_PATH = 'modelo_lsc_dinamico2.h5'  # Ajusta según tu ruta
    CLASSES_PATH = 'letras_dinamicas_classes (1).npy'  # Ajusta según tu ruta
    
    # Tamaño del buffer (número de frames para la secuencia)
    BUFFER_SIZE = 4  # Ajusta según lo que usaste en entrenamiento
    
    run_lsc_recognition(MODEL_PATH, CLASSES_PATH, BUFFER_SIZE)