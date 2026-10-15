# Vigilia

Sistema de visión artificial que detecta si un guardia se está quedando dormido o ya se durmió en una zona de vigilancia, y emite una alerta. Está pensado para un puesto de guardia delimitado con una o dos cámaras (por ejemplo, en un hospital).

Esta es una versión de producto mínimo viable: una sola aplicación Python, sin bases de datos ni contenedores, que funciona solo con CPU.

## Características

- Conexión a cámaras IP por RTSP, RTSPS y HTTP (MJPEG) con usuario y contraseña, con reconexión automática.
- Detección de personas con YOLO, seguimiento con ID persistente y estimación de 17 puntos del cuerpo.
- Análisis opcional de ojos cerrados y bostezos con MediaPipe.
- Heurística local de postura, movimiento y tiempo que se calcula en cada frame sin costo.
- Gemini Vision como juez principal, consultado solo cuando hay sospecha para reducir el uso de tokens.
- Funciona aunque el rostro no sea visible: en ese caso decide por postura, comportamiento y tiempo.
- Panel web con video en vivo, tabla de personas, alertas con evidencia fotográfica y alta de cámaras.
- Alerta de sistema si una cámara pierde la señal.

## Cómo funciona

```
Cámara (RTSP / RTSPS / HTTP)
   |
   v
camera.py       Captura con reconexión; siempre entrega el frame más reciente
   |
   v
optimizer.py    Limita los fps, reduce la resolución y mejora la imagen con poca luz
   |
   v
vision.py       YOLOv8n-pose: persona, ID (ByteTrack) y 17 puntos del cuerpo
                MediaPipe (opcional): ojos cerrados y bostezos
   |
   v
heuristics.py   Puntuación de 0 a 1 y estado local: despierta, somnolienta, dormida
   |
   v
gemini.py       Solo si hay sospecha: recortes pequeños de esa persona + resumen numérico
   |
   v
decision.py     Combina el estado local y el dictamen de Gemini
   |
   v
alerts.py       Alerta, foto anotada, registro, webhook opcional y reconocimiento
   |
   v
web.py          Panel web en tiempo real
```

### Criterios de la heurística local

La puntuación combina varias señales:

| Señal | Origen |
|---|---|
| Cabeza caída o ladeada respecto a los hombros | Puntos del cuerpo |
| Inclinación del tronco o cuerpo recostado | Puntos del cuerpo y proporción del cuadro |
| Cabeza no visible con hombros visibles | Puntos del cuerpo |
| Porcentaje de tiempo con los ojos cerrados | MediaPipe |
| Bostezos recientes | MediaPipe |
| Tiempo sin movimiento | Desplazamiento de los puntos |

La puntuación se suaviza en el tiempo y, si se mantiene alta durante varios segundos, el estado local pasa a somnolienta y luego a dormida. Ese estado local es lo que dispara la consulta a Gemini.

### Uso de Gemini

Gemini no recibe el video completo. Cuando la heurística sospecha, se envían unos 4 recortes de 384 píxeles de esa persona junto con un resumen numérico de los sensores locales, lo que equivale a unos 1.500 tokens por consulta. Hay límites para controlar el gasto:

- Mínimo de 20 segundos entre consultas por persona.
- Máximo de 120 consultas por hora (configurable).
- Pausa automática ante límite de cuota y desactivación ante clave inválida.

El panel muestra el total de consultas y tokens usados. Si Gemini no está disponible, el sistema sigue funcionando con la heurística local y tiempos de alerta más largos.

## Niveles de alerta

| Nivel | Significado | Quién lo decide |
|---|---|---|
| `somnolencia` | Se está quedando dormida | Gemini con confianza de 0.5 o más, o heurística local sostenida 20 s sin Gemini |
| `dormido` | Dormida | Gemini con confianza de 0.6 o más, o heurística local sostenida 30 s sin Gemini |
| `sistema` | Cámara sin señal por más de 60 s | Automático |

Si Gemini responde que la persona está despierta, no se genera alerta durante 60 segundos. Una alerta se repite cada 120 segundos mientras la situación persista y se marca como resuelta cuando la persona vuelve a estado normal.

## Requisitos

- Python 3.10 o superior.
- Equipo con CPU convencional; no requiere GPU.
- Cámara IP con acceso RTSP o HTTP, o un archivo de video para pruebas.
- Clave de API de Google AI Studio para activar Gemini (opcional pero recomendada).

## Instalación

```
git clone <url-del-repositorio>
cd <carpeta>
pip install -r requirements.txt
```

Copie la plantilla de configuración y complete la clave de Gemini:

```
copy .env.example .env      (Windows)
cp .env.example .env        (Linux / macOS)
```

Los pesos de YOLO y de MediaPipe se descargan automáticamente en la carpeta `models/` la primera vez que se ejecuta.

## Uso

```
python -m vigilia
```

Después abra el panel en el navegador: http://127.0.0.1:5000

### Agregar una cámara

Desde el panel, en la sección "Agregar / editar cámara", complete IP, usuario, contraseña y marca, y use "Probar conexión" antes de guardar. También puede definirla en el archivo `.env`:

```
CAMERA_1_NAME=Guardia pasillo
CAMERA_1_URL=rtsp://192.168.1.64:554/Streaming/Channels/102
CAMERA_1_USER=admin
CAMERA_1_PASSWORD=su_contraseña
```

Para trabajar con sub-streams de menor resolución (menos consumo de CPU), seleccione la marca de la cámara y se usará la ruta correspondiente. Marcas incluidas: Hikvision, Dahua, Tapo, Reolink, Uniview y Axis; para otras, use "genérico" e indique la ruta manualmente.

### Probar sin cámara

```
CAMERA_1_PROTOCOL=file
CAMERA_1_PATH=C:\ruta\al\video.mp4
```

### Zona de guardia

Para ignorar a personas que pasan fuera del puesto, defina un polígono con coordenadas normalizadas de 0 a 1 en el campo de zona de guardia del panel, por ejemplo:

```
[[0.1, 0.2], [0.9, 0.2], [0.9, 0.9], [0.1, 0.9]]
```

## Configuración

Todas las opciones se definen en `.env` y están documentadas en `.env.example`. Las principales:

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `GEMINI_API_KEY` | vacío | Clave de Google AI Studio |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Modelo a utilizar; verifique el nombre vigente |
| `GEMINI_MAX_CALLS_PER_HOUR` | 120 | Tope de consultas por hora |
| `INFERENCE_FPS` | 5 | Análisis por segundo con personas en zona |
| `IDLE_FPS` | 1.5 | Análisis por segundo con la zona vacía |
| `MAX_FRAME_SIDE` | 640 | Lado máximo del frame analizado |
| `FACE_ANALYSIS` | `auto` | Análisis facial local: `auto`, `on` u `off` |
| `SUSPECT_SECONDS` | 5 | Sospecha sostenida antes de consultar a Gemini |
| `SLEEP_SECONDS` | 12 | Sospecha fuerte sostenida para el estado local dormida |
| `MAX_CAMERAS` | 2 | Número máximo de cámaras |
| `WEB_HOST` / `WEB_PORT` | `127.0.0.1` / 5000 | Dirección del panel |
| `ALERT_WEBHOOK_URL` | vacío | URL que recibe cada alerta como JSON |

Los valores se validan al iniciar; si alguno es inválido, el programa lista todos los errores juntos y no arranca.

## Seguridad y privacidad

- El panel escucha solo en `127.0.0.1` por defecto. Para acceder desde otra máquina, `WEB_HOST=0.0.0.0` exige definir `DASHBOARD_USER` y `DASHBOARD_PASSWORD`. El panel no cifra el tráfico: use una red interna o un proxy con HTTPS.
- Las contraseñas de las cámaras no se devuelven por la API ni se escriben en los registros.
- El archivo `data/cameras.json` guarda las contraseñas en texto plano. Proteja la carpeta `data/` con los permisos del sistema operativo.
- Las imágenes de personas se envían a Gemini. Confirme con la institución la política de privacidad aplicable antes de usarlo en producción.
- No suba el archivo `.env` ni las carpetas `data/` y `models/` al repositorio; ya están en `.gitignore`.

## Calibración

Los umbrales de la heurística son un punto de partida. Antes de usarlo en producción se recomienda grabar unos minutos de la zona real con el guardia despierto, leyendo, usando el celular y cabeceando, y ajustar `SUSPECT_SECONDS`, `SLEEP_SECONDS` y la zona de guardia. La mejor precisión se obtiene con la cámara de frente al puesto, a la altura del pecho o ligeramente elevada y con buena iluminación.

## Estructura del proyecto

```
vigilia/
  __main__.py     Punto de entrada
  config.py       Configuración, validación y almacenamiento de cámaras
  camera.py       Conexión y captura de video
  optimizer.py    Reducción de resolución y mejora con poca luz
  vision.py       YOLO-pose con tracking y análisis facial
  heuristics.py   Puntuación de postura y estado local
  gemini.py       Cliente de Gemini Vision con control de gasto
  decision.py     Reglas de decisión final
  alerts.py       Alertas, evidencia y registro
  pipeline.py     Flujo completo por cámara
  render.py       Dibujo del cuadro, ID y esqueleto sobre el video
  app.py          Administración de cámaras en ejecución
  web.py          API y panel web
  static/         Interfaz del panel
tests/            Pruebas unitarias
data/             Cámaras, alertas y evidencia (se crea al ejecutar)
models/           Pesos descargados (se crea al ejecutar)
```

## Pruebas

```
python -m unittest discover -s tests
```

## Limitaciones conocidas

- Los umbrales no están calibrados con video de un entorno real.
- La integración con Gemini depende de una clave válida y de que el modelo configurado exista en su cuenta.
- Con el rostro oculto o la cámara desde un ángulo desfavorable, la precisión disminuye y la decisión depende más de la postura y el tiempo.
- No incluye autenticación de múltiples usuarios ni cifrado en el panel.
