# TelaScope – Roll Counter Dual-Cam

Sistema de conteo automático de rollos de tela usando **YOLO11n** fine-tuneado para una sola clase (`Roll`) y dos cámaras IP apuntando a la misma línea de ingreso, vistas desde ángulos distintos.

El objetivo es **contar de forma robusta y en tiempo real** los rollos que ingresan a un depósito, evitando doble conteo cuando un mismo rollo aparece en ambas cámaras.

---

## Características principales

- 🔍 **Detección** con modelo YOLO11n fine-tuneado para una sola clase: `Roll`.
- 🎥 **Soporte dual-cámara** (o videos de prueba) con:
  - Cam1 por defecto de **derecha a izquierda** (`R2L`)
  - Cam2 por defecto de **izquierda a derecha** (`L2R`)
- 🔁 **Fusión y deduplicación** de eventos entre cámaras:
  - Ventana temporal configurable para matchear un mismo rollo visto por ambas cámaras.
  - Si solo lo ve una cámara, igual se emite el evento luego de un `hold_window`.
- 🧠 **Tracking por ID** (Ultralytics + ByteTrack u otro tracker compatible).
- 📡 **Servidor TCP de eventos**:
  - Envío de eventos como JSON line-based (una línea JSON por evento).
  - Mensajes de tipo:
    - `CROSS` (cruce de barrera virtual)
    - `DONE` (rollo contado, con timestamp de entrada/salida y conteo global).
- 🌐 **Servidor HTTP (Flask)**:
  - Streaming MJPEG en `/cam1` y `/cam2`.
  - Página simple en `/` con ambas cámaras lado a lado.
- 🖥️ **UI local opcional** (`--show`):
  - Ventana con mosaico de ambas cámaras.
  - Panel inferior con:
    - Pedido (ejemplo hardcodeado `Pedido #A-12345`).
    - Hora actual.
    - Activos (en pantalla), pasados (conteo global).
    - Listado de tracks activos por cámara.
  - Botón de **Grabar / Detener** en esquina superior derecha.
- 🎬 **Grabación de video on-demand**:
  - Al habilitar grabación se genera un MP4 por cámara.
  - Salida en carpeta `runs_infer/...`.
- 📑 **Log unificado en CSV**:
  - Archivo `roll_events_fused.csv` con los eventos `DONE` ya fusionados.
- ⚙️ **Altamente configurable**:
  - Umbral de confianza, tamaño de imagen, device (`cpu`/GPU).
  - Geometría de las líneas virtuales y corredor vertical por cámara.
  - Ventanas temporales de matching / hold.
  - Puertos y host de los servidores TCP/HTTP.
  - Fuente TTF y tamaño para el panel de UI.

---

## Estructura del proyecto

Archivos principales:

- `cameras_roll_server.py`  
  Script principal. Maneja:
  - carga del modelo,
  - tracking por cámara,
  - lógica de cruce de barreras,
  - fusión de eventos,
  - UI local,
  - servidores TCP y HTTP.

- `best.pt`  
  Pesos del modelo YOLO11n fine-tuneado para la clase `Roll`.  
  El script también permite usar un fallback (`yolo11n.pt`) si no encuentra `best.pt`.

- `cam1-6seg.mp4`, `cam2-6seg.mp4`  
  Videos cortos de prueba (6 segundos) que simulan las cámaras reales.

- `requirements.txt`  
  Dependencias de Python.

---

## Instalación

Se recomienda Python 3.10+ (al menos 3.8).

```bash
git clone https://github.com/aurebidart/TelaScope.git
cd TelaScope

python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
````

> El script internamente usa, entre otros:
>
> * `ultralytics` (YOLO)
> * `opencv-python`
> * `numpy`
> * `flask` + `werkzeug`
> * `Pillow`
> * `turbojpeg` (opcional, para codificación JPEG más rápida)

---

## Uso rápido con los videos de prueba

Con UI local (+ panel + botón de grabar):

```bash
python cameras_roll_server.py \
  --source1 cam1-6seg.mp4 --dir1 R2L \
  --source2 cam2-6seg.mp4 --dir2 L2R \
  --show
```

Sin UI local (solo servidores HTTP y TCP):

```bash
python cameras_roll_server.py \
  --source1 cam1-6seg.mp4 --dir1 R2L \
  --source2 cam2-6seg.mp4 --dir2 L2R
```

Al iniciar, el programa imprime algo como:

* Puerto HTTP: `http://localhost:<puerto>/`
* Puerto TCP de eventos: `<puerto>`

---

## Uso con cámaras IP reales

Ejemplo genérico con RTSP (cam1 R2L, cam2 L2R):

```bash
python cameras_roll_server.py ^
  --source1 "rtsp://admin:<password>@192.168.36.213:554/cam/realmonitor?channel=1&subtype=0" --dir1 R2L ^
  --source2 "rtsp://admin:<password>@192.168.36.214:554/cam/realmonitor?channel=1&subtype=0" --dir2 L2R ^
  --left_ratio1 0.35 --right_ratio1 0.70 --band_top_ratio1 0.0 --band_bot_ratio1 1.0 --band_slack_px1 20 ^
  --left_ratio2 0.40 --right_ratio2 0.70 --band_top_ratio2 0.0 --band_bot_ratio2 1.0 --band_slack_px2 20 ^
  --conf 0.30 --show
```

Ajustá:

* URLs RTSP / HTTP / RTMP de tus cámaras.
* Dirección (`--dir1`, `--dir2`) según el movimiento real de los rollos.
* Ratios de líneas y corredor según tu encuadre.

---

## Argumentos principales

Para ver la ayuda completa:

```bash
python cameras_roll_server.py --help
```

Los más importantes:

### Fuentes

* `--source1` (requerido): ruta o URL de **Cam1**
  Soporta:

  * archivos de video (`.mp4`, etc.)
  * streams `rtsp://`, `http://`, `rtmp://`, `udp://`, `tcp://`
* `--source2` (requerido): ruta o URL de **Cam2**
* `--dir1` (`R2L` por defecto): dirección de flujo de Cam1 (`L2R` o `R2L`)
* `--dir2` (`L2R` por defecto): dirección de flujo de Cam2

### Modelo / tracking

* `--conf` (float, default `0.30`): umbral de confianza para detecciones.
* `--img` (int, default `768`): tamaño de imagen de entrada.
* `--device`: `cpu` o ID de GPU (ej. `0`).
* `--classes`: lista de IDs de clases a usar (por defecto todas; en este proyecto suele ser una sola clase `Roll`).
* `--weights_glob` (default `best.pt`): patrón para localizar pesos.
* `--fallback_weights` (default `yolo11n.pt`): modelo fallback.
* `--tracker` (default `bytetrack.yaml`): config de tracker Ultralytics.
* `--min_track_len`: frames mínimos antes de validar entrada.

### Visualización / rendimiento

* `--show`: habilita la ventana local con mosaico + panel + botón de grabar.
* `--out`: carpeta base para videos grabados (por defecto se usa `runs_infer/...`).
* `--fps_cap`: límite de FPS de procesamiento/muestreo (0 = sin límite).
* `--slowdown`: factor de ralentización (>= 1.0).
* `--active_timeout`: segundos sin ver un track para ocultarlo/purgarlo del panel/UI.

### Geometría / líneas virtuales

Las líneas y el “corredor” vertical se definen como fracciones de ancho/alto de la imagen.
Hay valores globales y overrides por cámara.

Global:

* `--left_ratio`, `--right_ratio`: posición X de las dos barreras verticales (0–1).
* `--band_top_ratio`, `--band_bot_ratio`: límites superior/inferior del corredor (0–1).
* `--band_slack_px`: margen extra vertical en píxeles.

Overrides específicos:

* Cam1: `--left_ratio1`, `--right_ratio1`, `--band_top_ratio1`, `--band_bot_ratio1`, `--band_slack_px1`
* Cam2: `--left_ratio2`, `--right_ratio2`, `--band_top_ratio2`, `--band_bot_ratio2`, `--band_slack_px2`

### Fusión de cámaras

* `--match_window` (segundos): Δt máximo entre detecciones de ambas cámaras para considerarlas el mismo rollo.
* `--hold_window` (segundos): tiempo máximo que se espera por la otra cámara antes de emitir evento single-cam.

### Servidores

* `--host` (default `0.0.0.0`): host donde se bindean HTTP y TCP.
* `--http_port` (default `0`): puerto HTTP (0 = puerto aleatorio libre).
* `--event_port` (default `0`): puerto TCP de eventos (0 = puerto aleatorio libre).

---

## Eventos por socket TCP

Al iniciar, se levanta un servidor TCP (por defecto en un puerto aleatorio que se imprime en consola).

* Cada cliente que se conecta recibe un mensaje de bienvenida:

```json
{"type":"HELLO","info":"Eventos de rollos (DONE). Una línea JSON por evento."}
```

* Luego, el servidor envía eventos como **JSON por línea**:

### Evento `CROSS`

Se emite cuando un track cruza una de las líneas virtuales:

```json
{
  "type": "CROSS",
  "camera": "cam1",       // "cam1" o "cam2"
  "which": "LEFT",        // "LEFT" o "RIGHT"
  "track_id": 7,
  "ts_ms": 1711122334455, // timestamp ms (epoch aproximadamente)
  "frame": 123            // frame aproximado
}
```

### Evento `DONE`

Se emite una sola vez por rollo real (ya fusionado entre cámaras):

```json
{
  "type": "DONE",
  "id": 12,               // ID incremental del rollo
  "enter": 1711122334000, // ms desde epoch ~ momento de entrada
  "exit": 1711122339500,  // ms desde epoch ~ momento de salida
  "count": 1,
  "total": 42,            // conteo global acumulado
  "cams": ["cam1","cam2"] // cámaras que lo vieron (1 o 2)
}
```

> Si solo una cámara ve el rollo dentro de `hold_window`, igualmente se emite `DONE` con `cams` de 1 sola entrada.

---

## Streaming HTTP

El servidor HTTP (Flask) expone:

* `GET /`
  Página HTML sencilla con Cam1 y Cam2 lado a lado.

* `GET /cam1`
  Stream MJPEG con la vista de Cam1 (incluyendo HUD, bbox, líneas, etc.).

* `GET /cam2`
  Stream MJPEG con la vista de Cam2.

Al iniciar, se imprime algo como:

```text
[SERVER] HTTP (Flask) escuchando en 0.0.0.0:<port>
[SERVER] Abrí http://localhost:<port>/ para ver ambos streams
```

---

## UI local y grabación

Con `--show`:

* Se abre una ventana llamada:
  `Roll Counter – Dual + Panel 2 Columnas`
* Parte superior:

  * Mosaico Cam1 + Cam2.
  * Botón **● Grabar / ■ Detener** arriba a la derecha.
* Parte inferior:

  * Panel con:

    * Pedido (texto fijo de ejemplo).
    * Hora del sistema.
    * Cuenta de rollos **activos** (en tracking).
    * Cuenta de rollos **pasados** (global fused).
    * Listas de tracks activos por cámara, con:

      * ID
      * Última posición `cx, cy`
      * timestamps de cruce de líneas.

Controles:

* `ESC` o `q`: cerrar UI y apagar todo ordenadamente.
* Click en el botón **● Grabar / ■ Detener**: iniciar/detener grabación simultánea de ambas cámaras.
* Tecla `r`: toggle de grabación (fallback si el backend de OpenCV no soporta mouse callback).

Videos grabados:

* Se guardan en `runs_infer/...` (o en la carpeta indicada con `--out`), con nombre tipo:

  * `cam1_output.mp4`
  * `cam2_output.mp4`

---

## CSV de salida (eventos fusionados)

En cada ejecución se crea una carpeta de salida, por ejemplo:

```text
runs_infer/dual_YYYYMMDD_HHMMSS/
```

Dentro, se genera:

```text
roll_events_fused.csv
```

Con encabezado:

```csv
roll_id,cams,t_fused,t_cam1_start,t_cam1_end,t_cam2_start,t_cam2_end,dur_cam1_s,dur_cam2_s,count_global
```

Cada fila corresponde a un `DONE`:

* `roll_id`: ID incremental del rollo.
* `cams`: `"cam1"`, `"cam2"` o `"cam1&cam2"`.
* `t_fused`: timestamp promedio de pasaje (en formato `hh:mm:ss.mmm`).
* `t_camX_start`, `t_camX_end`: timestamps de cruce de las líneas en cada cámara.
* `dur_camX_s`: duración (en segundos) del pasaje según cada cámara.
* `count_global`: valor del contador global luego de contar ese rollo.

---

## Notas de desarrollo

* En modo **sin UI** (sin `--show`), el proceso corre en background solo con:

  * servidor HTTP,
  * servidor de eventos TCP,
  * CSV.
* La lógica intenta reconectar/reiniciar el loop de inferencia si el stream se corta o termina.
* Para producción se recomienda:

  * fijar `--http_port` y `--event_port`,
  * loguear stdout/stderr del proceso,
  * usar supervisión externa (systemd, docker, etc.).

