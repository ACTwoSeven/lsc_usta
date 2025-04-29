# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 08:56:21 2025

@author: Andre
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Input, Masking, LSTM, Dropout, Dense
from tensorflow.keras.models import Model, load_model
import os

class LSCRecognitionSystem:
    """Sistema de reconocimiento para Lengua de Señas Colombiana que combina
    reconocimiento estático y dinámico."""

    def __init__(self,
                 static_model_path='best_model_landmarks.h5',
                 dynamic_model_path='modelo_lsc_dinamico2.h5',
                 dynamic_classes_path='letras_dinamicas_classes.npy',
                 buffer_size=4,
                 auto_mode=True):

        self.buffer_size = buffer_size
        self.confidence_threshold = 0.7
        self.static_mode = True  # Iniciar en modo estático
        self.auto_mode = auto_mode  # Modo automático de detección

        # Variables para detección automática de movimiento
        self.position_history = []
        self.max_history = 15  # Aumentado de 10 a 15 para mejor análisis
        self.movement_threshold = 0.015  # Reducido de 0.03 a 0.015
        self.movement_count = 0  # Contador para movimiento sostenido
        self.last_mode_change_time = 0  # Para controlar frecuencia de cambios

        # Variables para estabilización de predicción
        self.stable_letter = ""
        self.stable_count = 0
        self.required_stable_frames = 7  # Número de frames requeridos para considerar una letra estable
        self.tiempo_ultima_deteccion = 0 #para el intervalo de tiempo entre letras

        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Inicializar buffer para secuencias dinámicas
        self.buffer = deque(maxlen=buffer_size)

        # Verificar existencia de archivos de modelo
        self._check_model_files(static_model_path, dynamic_model_path, dynamic_classes_path)

        # Cargar modelo estático
        print("Cargando modelo estático...")
        self.static_model = self._load_static_model(static_model_path)
        self.static_classes = ['a','b','c','d','e','f','i','k','l','m','n','o','p','q','r','t','u','v','w','x','y']
        print(f"Modelo estático cargado. Clases: {self.static_classes}")

        # Cargar modelo dinámico
        print("Cargando modelo dinámico...")
        self.dynamic_model, self.dynamic_classes = self._load_dynamic_model(
            dynamic_model_path, dynamic_classes_path)
        print(f"Modelo dinámico cargado. Clases: {self.dynamic_classes}")

        # Verificar modelos
        self.verify_models()

        # Variables para mostrar predicciones
        self.current_letter = ""
        self.prediction_confidence = 0.0

    def _check_model_files(self, static_path, dynamic_path, classes_path):
        """Verifica que los archivos de modelo existan."""
        missing_files = []

        if not os.path.exists(static_path):
            missing_files.append(f"Modelo estático: {static_path}")

        if not os.path.exists(dynamic_path):
            missing_files.append(f"Modelo dinámico: {dynamic_path}")

        if not os.path.exists(classes_path):
            missing_files.append(f"Clases dinámicas: {classes_path}")

        if missing_files:
            print("¡ADVERTENCIA! No se encontraron los siguientes archivos:")
            for file in missing_files:
                print(f"  - {file}")
            print("\nPor favor, asegúrate de que estos archivos estén en el directorio correcto.")

    def verify_models(self):
        """Verifica que los modelos estén cargados correctamente."""
        models_ok = True

        # Verificar modelo estático
        try:
            # Prueba simple con datos aleatorios
            test_input = np.random.random((1, 63))
            test_output = self.static_model.predict(test_input, verbose=0)
            print(f"✓ Modelo estático OK - Clases: {self.static_classes}")
        except Exception as e:
            print(f"✗ Error en modelo estático: {e}")
            models_ok = False

        # Verificar modelo dinámico
        try:
            # Prueba simple con datos aleatorios
            test_seq = np.random.random((1, self.buffer_size, 63))
            test_output = self.dynamic_model.predict(test_seq, verbose=0)
            print(f"✓ Modelo dinámico OK - Clases: {self.dynamic_classes}")
        except Exception as e:
            print(f"✗ Error en modelo dinámico: {e}")
            models_ok = False

        return models_ok

    def _load_static_model(self, model_path):
        """Carga el modelo estático."""
        try:
            # Intentar cargar directamente
            return load_model(model_path)
        except:
            try:
                # Si falla, intentar con custom_objects
                return load_model(model_path, custom_objects={'NotEqual': tf.math.not_equal})
            except Exception as e:
                print(f"Error al cargar modelo estático: {e}")
                # Crear modelo dummy para no romper el sistema
                print("Creando modelo estático dummy...")
                inp = Input(shape=(63,))
                x = Dense(64, activation='relu')(inp)
                out = Dense(len(self.static_classes), activation='softmax')(x)
                model = Model(inp, out)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
                return model

    def _load_dynamic_model(self, model_path, classes_path):
        """Recrea y carga el modelo dinámico."""
        try:
            # Cargar clases
            if os.path.exists(classes_path):
                classes = np.load(classes_path)
            else:
                print(f"Archivo de clases no encontrado: {classes_path}")
                # Clases dinámicas de ejemplo
                classes = np.array(['j', 'z', 'ñ', 'h', 'g'])

            num_classes = len(classes)

            # Recrear arquitectura
            input_dim = 63  # 21 landmarks x 3 coordinates

            inp = Input(shape=(None, input_dim))
            x = Masking(mask_value=0.0)(inp)
            x = LSTM(128, return_sequences=True)(x)
            x = Dropout(0.3)(x)
            x = LSTM(64)(x)
            x = Dropout(0.3)(x)
            x = Dense(64, activation='relu')(x)
            out = Dense(num_classes, activation='softmax')(x)

            model = Model(inp, out)
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Intentar cargar pesos si existe el archivo
            if os.path.exists(model_path):
                try:
                    model.load_weights(model_path)
                except:
                    # Si falla, intentar cargar completo y extraer pesos
                    try:
                        temp_model = load_model(model_path, compile=False,
                                                 custom_objects={'NotEqual': tf.math.not_equal})
                        temp_weights = '/tmp/temp_weights.h5'
                        temp_model.save_weights(temp_weights)
                        model.load_weights(temp_weights)
                    except Exception as e:
                        print(f"Error al cargar pesos del modelo dinámico: {e}")
            else:
                print(f"Archivo de modelo dinámico no encontrado: {model_path}")

            return model, classes
        except Exception as e:
            print(f"Error al cargar modelo dinámico: {e}")
            # Crear modelo dummy para no romper el sistema
            print("Creando modelo dinámico dummy...")
            inp = Input(shape=(None, 63))
            x = LSTM(64)(inp)
            out = Dense(5, activation='softmax')(x)  # 5 clases dummy
            model = Model(inp, out)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            return model, np.array(['j', 'z', 'ñ', 'h', 'g'])  # Clases dummy

    def preprocess_static_landmarks(self, landmarks):
        """Preprocesa landmarks para el modelo estático."""
        lm = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        # centrar respecto a la muñeca (punto 0)
        lm[:, :2] -= lm[0, :2]
        # escalar para que la dimensión máxima del plano xy sea 1
        scale = np.max(np.linalg.norm(lm[:, :2], axis=1))
        if scale > 0:
            lm[:, :2] /= scale
        return lm.flatten()[np.newaxis, :]  # shape (1,63)

    def extract_dynamic_landmarks(self, landmarks):
        """Extrae landmarks para el modelo dinámico."""
        lm = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        # centrar en muñeca
        lm[:, :2] -= lm[0, :2]
        # escalar para que la distancia máxima en XY sea 1
        scale = np.max(np.linalg.norm(lm[:, :2], axis=1))
        if scale > 0:
            lm[:, :2] /= scale
        return lm.flatten()

    def predict_static(self, landmarks):
        """Realiza predicción con el modelo estático."""
        x = self.preprocess_static_landmarks(landmarks)
        probs = self.static_model.predict(x, verbose=0)[0]
        idx = np.argmax(probs)
        letter = self.static_classes[idx]
        conf = probs[idx]
        return letter, conf

    def update_dynamic_buffer(self, landmarks=None):
        """Actualiza el buffer para modelo dinámico."""
        if landmarks is not None:
            vec = self.extract_dynamic_landmarks(landmarks)
            self.buffer.append(vec)
        else:
            # Si no hay mano visible, repetir último vector o añadir ceros
            if len(self.buffer) > 0:
                self.buffer.append(self.buffer[-1])
            else:
                self.buffer.append(np.zeros(63, dtype=np.float32))

    def predict_dynamic(self):
        """Realiza predicción con el modelo dinámico."""
        if len(self.buffer) == self.buffer_size:
            seq = np.stack(self.buffer)[np.newaxis, ...]  # (1, buffer_size, 63)
            probs = self.dynamic_model.predict(seq, verbose=0)[0]
            idx = np.argmax(probs)
            letter = self.dynamic_classes[idx]
            conf = probs[idx]
            return letter, conf
        return None, 0.0

    def toggle_mode(self):
        """Alterna entre modo estático y dinámico."""
        self.static_mode = not self.static_mode
        # Limpiar buffer al cambiar de modo
        self.buffer.clear()
        self.current_letter = ""
        self.prediction_confidence = 0.0
        self.stable_letter = "" #reset stable letter
        self.stable_count = 0 #reset stable count
        return self.static_mode

    def detect_movement(self, landmarks):
        """Detecta si hay movimiento significativo en la mano con lógica mejorada."""
        if not landmarks:
            return False

        # Extraer posición actual (centro de la palma, usando puntos clave específicos)
        palm_center = np.mean([[landmarks[0].x, landmarks[0].y],  # Muñeca
                              [landmarks[5].x, landmarks[5].y],  # Base del pulgar
                              [landmarks[9].x, landmarks[9].y],  # Base del dedo medio
                              [landmarks[13].x, landmarks[13].y],  # Base del anular
                              [landmarks[17].x, landmarks[17].y]  # Base del meñique
                              ], axis=0)

        # Posiciones de puntas de dedos para detectar gestos
        finger_tips = np.array([[landmarks[8].x, landmarks[8].y],  # Punta índice
                                 [landmarks[12].x, landmarks[12].y],  # Punta medio
                                 [landmarks[16].x, landmarks[16].y],  # Punta anular
                                 [landmarks[20].x, landmarks[20].y]])  # Punta meñique

        # Guardar en historial
        self.position_history.append(palm_center)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

        # Si no tenemos suficientes puntos, no podemos determinar movimiento
        if len(self.position_history) < 3:  # Reducido a 3 para detección más rápida
            return False

        # Calcular varianza del movimiento de la palma
        positions = np.array(self.position_history)
        palm_variance = np.sum(np.var(positions, axis=0))

        # Calcular velocidad del movimiento (distancia entre frames consecutivos)
        if len(positions) >= 2:
            velocities = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
            avg_velocity = np.mean(velocities)
        else:
            avg_velocity = 0

        # También detectar movimiento en las puntas de los dedos
        # (importante para letras como J, Z que tienen movimiento principalmente en dedos)
        if hasattr(self, 'prev_finger_tips'):
            finger_movement = np.mean(np.linalg.norm(finger_tips - self.prev_finger_tips, axis=1))
        else:
            finger_movement = 0

        # Guardar posición actual para próxima comparación
        self.prev_finger_tips = finger_tips

        # Combinación de factores para determinar si hay movimiento
        is_moving = (palm_variance > self.movement_threshold or
                     avg_velocity > self.movement_threshold * 1.5 or
                     finger_movement > self.movement_threshold * 2)

        return is_moving

    def auto_detect_mode(self, landmarks):
        """Determina automáticamente si usar modo estático o dinámico con lógica mejorada."""
        if not landmarks:
            return

        is_moving = self.detect_movement(landmarks)

        # Añadir tiempo mínimo en cada modo para evitar cambios rápidos
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # Inicializar tiempo de cambio si no existe
        if not hasattr(self, 'last_mode_change_time'):
            self.last_mode_change_time = current_time

        # Tiempo mínimo que debe transcurrir para cambiar de modo (en segundos)
        min_time_in_mode = 0.7  # 700ms

        # Registrar si se ha detectado movimiento consistente
        if is_moving:
            self.movement_count += 1
        else:
            self.movement_count = max(self.movement_count - 1, 0)

        # Para limitar el contador
        self.movement_count = min(self.movement_count, 10)

        time_elapsed = current_time - self.last_mode_change_time

        # Cambiar a modo dinámico si:
        # - Estamos en modo estático
        # - Se detecta movimiento consistente (contador > umbral)
        # - Ha pasado suficiente tiempo desde el último cambio
        if self.static_mode and self.movement_count > 3 and time_elapsed > min_time_in_mode:
            print("Cambiando a modo DINÁMICO (movimiento detectado)")
            self.static_mode = False
            self.buffer.clear()  # Reiniciar buffer al cambiar a dinámico
            self.last_mode_change_time = current_time
            self.stable_letter = ""  # Reset stable letter
            self.stable_count = 0  # Reset stable count
        # Cambiar a modo estático si:
        # - Estamos en modo dinámico
        # - No se detecta movimiento por un tiempo
        # - Ha pasado suficiente tiempo desde el último cambio
        elif not self.static_mode and self.movement_count == 0 and time_elapsed > min_time_in_mode:
            # Verificar que realmente no hay movimiento revisando las posiciones recientes
            if len(self.position_history) >= 3:
                recent_positions = np.array(self.position_history[-3:])
                recent_variance = np.sum(np.var(recent_positions, axis=0))

                if recent_variance < self.movement_threshold * 0.4:  # Umbral más bajo para volver a estático
                    print("Cambiando a modo ESTÁTICO (sin movimiento)")
                    self.static_mode = True
                    self.last_mode_change_time = current_time
                    self.stable_letter = ""  # Reset stable letter
                    self.stable_count = 0  # Reset stable count

    def run(self):
        """Ejecuta el sistema de reconocimiento en tiempo real."""
        # Configurar cámara
        print("Iniciando cámara...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

        print("Sistema LSC iniciado!")
        print("- Presiona 'A' para activar/desactivar modo automático")
        print("- Presiona 'M' para cambiar de modo manualmente (estático/dinámico)")
        print("- Presiona 'ESC' para salir")

        # Para mostrar mensajes de estado temporales
        status_message = ""
        status_time = 0

        palabra_actual = ""
        ultima_letra_detectada = ""
        intervalo_nueva_letra = 3.0  # Intervalo mínimo para una nueva letra (o la misma)
        intervalo_misma_letra = 2.5 # Intervalo mínimo para detectar la misma letra de nuevo

        tiempo_actual = cv2.getTickCount() / cv2.getTickFrequency() # Inicializar tiempo_actual

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame de la cámara")
                break

            # Voltear horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)

            # Procesar frame con MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # Preparar panel de información
            frame_height, frame_width = frame.shape[:2]
            info_panel = np.zeros((frame_height, 250, 3), dtype=np.uint8)

            # Detectar y procesar mano
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Dibujar landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Detectar modo automáticamente si está activado
                if self.auto_mode:
                    self.auto_detect_mode(hand_landmarks.landmark)

                # Procesar según el modo
                letra_predicha = None
                confianza_prediccion = 0.0

                if self.static_mode:
                    # Modo estático
                    letter, conf = self.predict_static(hand_landmarks.landmark)
                    if conf >= 0.90: #mayor confianza
                         letra_predicha = letter
                         confianza_prediccion = conf
                else:
                    # Modo dinámico
                    self.update_dynamic_buffer(hand_landmarks.landmark)
                    letter, conf = self.predict_dynamic()
                    if letter and conf >= 0.90: #mayor confianza
                        letra_predicha = letter
                        confianza_prediccion = conf

                # Lógica de estabilización de letra
                if letra_predicha:
                    if letra_predicha == self.stable_letter:
                        self.stable_count += 1
                    else:
                        self.stable_letter = letra_predicha
                        self.stable_count = 1

                    if self.stable_count >= self.required_stable_frames:
                        # Considerar la letra estable y agregarla
                        tiempo_actual = cv2.getTickCount() / cv2.getTickFrequency()
                        puede_agregar = False

                        if letra_predicha != ultima_letra_detectada:
                            if (tiempo_actual - self.tiempo_ultima_deteccion) > intervalo_nueva_letra:
                                puede_agregar = True
                        else:
                            if (tiempo_actual - self.tiempo_ultima_deteccion) > intervalo_misma_letra:
                                puede_agregar = True

                        if puede_agregar:
                            palabra_actual += letra_predicha
                            ultima_letra_detectada = letra_predicha
                            self.tiempo_ultima_deteccion = tiempo_actual
                            print(f"Palabra actual: {palabra_actual}")

                        self.current_letter = letra_predicha
                        self.prediction_confidence = confianza_prediccion
                    else:
                        # Esperar a que la letra se estabilice
                        self.current_letter = ""
                        self.prediction_confidence = 0.0
                else:
                    # No hay letra predicha, resetear estado de estabilización
                    self.stable_letter = ""
                    self.stable_count = 0
                    self.current_letter = ""
                    self.prediction_confidence = 0.0

                # Indicadores de movimiento para depuración (si está en auto_mode)
                if self.auto_mode:
                    # Detectar movimiento para visualización
                    is_moving = self.detect_movement(hand_landmarks.landmark)

                    # Mostrar valor de varianza y contador de movimiento
                    if hasattr(self, 'position_history') and len(self.position_history) >= 2:
                        positions = np.array(self.position_history)
                        variance = np.sum(np.var(positions, axis=0))

                        cv2.putText(info_panel, f"Var: {variance:.4f}", (10, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0) if variance > self.movement_threshold else (150, 150, 150), 1)

                    # Mostrar contador de movimiento
                    if hasattr(self, 'movement_count'):
                        cv2.putText(info_panel, f"Mov: {self.movement_count}", (10, 270),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0) if self.movement_count > 3 else (150, 150, 150), 1)

                    # Indicador visual de detección de movimiento
                    move_indicator_color = (0, 255, 0) if is_moving else (0, 0, 255)
                    cv2.circle(info_panel, (220, 240), 10, move_indicator_color, -1)
            else:
                # Si no hay mano visible, resetear el tiempo de última detección y el estado de estabilización
                ultima_letra_detectada = ""
                self.tiempo_ultima_deteccion = tiempo_actual
                self.stable_letter = ""
                self.stable_count = 0
                self.current_letter = ""
                self.prediction_confidence = 0.0

            # Dibujar información en el panel
            mode_text = "ESTÁTICO" if self.static_mode else "DINÁMICO"
            cv2.putText(info_panel, f"Modo: {mode_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.static_mode else (0, 165, 255), 2)
            cv2.putText(info_panel, "Letra:", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Mostrar letra con confianza
            if self.current_letter:
                cv2.putText(info_panel, self.current_letter.upper(), (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                            (0, 255, 0) if self.prediction_confidence > 0.8 else (0, 200, 200),
                            3)

                cv2.putText(info_panel, f"Conf: {self.prediction_confidence:.2f}", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            else:
                cv2.putText(info_panel, "---", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 200), 3)

            # Mostrar instrucciones y estado de auto-modo
            auto_status = "ON" if self.auto_mode else "OFF"
            cv2.putText(info_panel, f"Auto: {auto_status}", (10, frame_height - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if self.auto_mode else (150, 150, 150), 1)
            cv2.putText(info_panel, "A: Activar/desactivar auto", (10, frame_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(info_panel, "M: Cambiar modo manual", (10, frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            if not self.static_mode:
                # Mostrar estado del buffer en modo dinámico
                cv2.putText(info_panel, f"Buffer: {len(self.buffer)}/{self.buffer_size}",
                            (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Mostrar mensaje de estado temporal si existe
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - status_time < 2.0 and status_message:  # Mostrar por 2 segundos
                cv2.putText(info_panel, status_message, (10, frame_height - 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Mostrar la palabra actual en el panel
            cv2.putText(info_panel, f"Palabra: {palabra_actual}", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Combinar frame y panel de información
            combined_frame = np.hstack((frame, info_panel))

            # Mostrar el frame combinado
            cv2.imshow('Sistema LSC', combined_frame)

            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC para salir
                break
            elif key == ord('m') or key == ord('M'):  # M para cambiar modo manualmente
                self.auto_mode = False  # Desactivar auto al cambiar manualmente
                new_mode = "ESTÁTICO" if self.toggle_mode() else "DINÁMICO"
                status_message = f"Modo cambiado a: {new_mode}"
                status_time = current_time
                print(f"Modo cambiado manualmente a: {new_mode}")
            elif key == ord('a') or key == ord('A'):  # A para toggle auto-modo
                self.auto_mode = not self.auto_mode
                status_message = f"Modo automático: {'ACTIVADO' if self.auto_mode else 'DESACTIVADO'}"
                status_time = current_time
                print(f"Modo automático: {'ACTIVADO' if self.auto_mode else 'DESACTIVADO'}")
            elif key == ord('b') or key == ord('B') or key == 8:  # B o Retroceso para borrar
                if palabra_actual:
                    palabra_actual = palabra_actual[:-1]
                    print(f"Palabra actual (borrado): {palabra_actual}")

        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("Sistema finalizado")


# Ejemplo de uso
if __name__ == "__main__":
    try:
        # Configurar rutas a los modelos
        STATIC_MODEL_PATH = 'model_lsc_static.h5'
        DYNAMIC_MODEL_PATH = 'model_lsc_dinamic.h5'
        DYNAMIC_CLASSES_PATH = 'dinamic_classes.npy'

        # Crear y ejecutar el sistema
        system = LSCRecognitionSystem(
            static_model_path=STATIC_MODEL_PATH,
            dynamic_model_path=DYNAMIC_MODEL_PATH,
            dynamic_classes_path=DYNAMIC_CLASSES_PATH,
            buffer_size=4,
            auto_mode=True  # Activar modo automático por defecto
        )

        system.run()

    except Exception as e:
        print(f"Error al iniciar el sistema: {e}")
