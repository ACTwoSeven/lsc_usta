# lsc_usta (Lengua de señas colombiana)
Proyecto universitario integrando lengua de señas colombiana. Forma palabras con el abecedario del LSC.

Este proyecto fue creado en la Universidad Santo Tomás de Bucaramanga, en el programa de Ingeniería de Telecomunicaciones para la clase de Python.

# Dataset
Para desarrollar este proyecto se utilizó un dataset que estaba conformado por todas las letras del abecedario, a-z, incluido las letras dinámicas (g, h, j, ñ, rr, s, z)
En todo el dataset utilizamos imágenes y videos propios de los estudiantes de la clase de python 2025, junto a un dataset de open access perteneciente a la Universidad del Cauca con DOI https://doi.org/10.1016/j.dib.2024.111213. En el dataset de la Universidad del Cauca tienen +200 imágenes por cada letra, por lo que nos funciona para complementar nuestro dataset y poder fortalecer el entrenamiento. 
Dataset Universidad del Cauca: 
![image](https://github.com/user-attachments/assets/9336cfe8-586f-4a35-8adb-f4a6fd369eae)

Dataset Universidad Santo Tomás Bucaramanga: ![image](https://github.com/user-attachments/assets/5120544a-ac5b-45d6-9f37-866ea1fbffa9)

Para nuestro dataset aprovechamos el curso gratuito de Academy Edutin y aprendimos sobre LSC, por lo cual aplicamos letras como rr que hace parte del abecedario lsc.
En la carpeta de dataset vas a poder encontrar tanto el dinámico como estático. Esta división se debe al entrenamiento, pues por un lado entrenamos un modelo con el dataset de las letras estáticas y por otro lado el de letras dinámicas.
Todas las imágenes tienen un tamaño de 640x480.
En las letras estáticas se hizo data augmentation para generalizar los datos, luego por cada imagen se sacaron los landmarks, en total 21 de cada mano por cada posición x,y,z teniendo un vector de 63 puntos guardados en archivos npy para entrenar el modelo estático.
En las letras dinámicas se sacaron hasta 4 frames por video, luego se hizo una secuencia en archivos npy con los landmarks generado por mediapipe para entrenar el modelo.

# Modelos
Para el modelo estático se uso una división de los datos 70/15/15, donde 70% fue usado para el entrenamiento, 15% para validación y 15% para testeo. En total se usaron 469 imágenes para cada letra, generando el archivo npy de los vectores. 
![image](https://github.com/user-attachments/assets/f2756d18-729d-4221-9ba1-cb493f8025d3)
![image](https://github.com/user-attachments/assets/b81fd7c6-efac-4ff3-b12c-7dd2b6eeea25)
Al usar data augmentation los resultados de la métrica accuracy no son muy altos, sin embargo, al llevar el modelo a la práctica sin duda responde muy bien a las letras.

Para el modelo dinámico en cambio se uso un 80/10/10, con un 80% para entrenamiento y una menor cantidad de datos, en este caso al juntar 4 frames para cada frecuencia logramos generar hasta 32 secuencias para una letra, secuencias que nos ayudan a entrenar el modelo obteniendo:
![image](https://github.com/user-attachments/assets/de7c7454-430c-4eab-8d45-469ff17e7bfc)
![image](https://github.com/user-attachments/assets/7372a7b3-5e2b-4b5a-a09a-58b533a278c7)
En este caso se repite el mismo patrón, donde el accuracy no es el mejor y se podría optimizar el modelo, sin embargo, en la práctica el comportamiento es más que adecuado para los recursos gastados y el fin del modelo.

# Resultado
![image](https://github.com/user-attachments/assets/9db892ae-9183-4327-8f77-4fa9ff694b03)
En la app programada por python se integraron los modelos, principalmente funcionan con un sistema de imágenes tomadas en secuencia, para comprobar si hay movimiento o son letras estáticas. Con esa lógica se van escribiendo las palabras poco a poco, también cuenta con una función para usar solo el modo dinámico o estático que cuenta con mejor accuracy de forma individual, además, contamos con la letra b que nos ayuda a borrar alguna letra que no deseemos.

# Recomendaciones
Se recomienda para futuros trabajos trabajar con el dataset de letras en movimiento, acomodando de forma más uniforme los frames de cada video y así poder generar mejor las secuencias de landmarks. También se podría integrar una librería que ayuda a completar las palabras y hacer el proceso más rápido.
