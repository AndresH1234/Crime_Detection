import cv2
import csv
import os

data_folder = os.path.join(os.path.dirname(__file__), "../data/Anomaly-Detection-Dataset")

# Lista de eventos
Events = ['Abuse', 'Arson', 'Assault', 'Burglary', 'Fighting', 
        'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

#Events = ['Assault', 'Burglary', 'Robbery', 'Shoplifting', 'Stealing', 'Vandalism']

# Función para obtener los nombres de los videos dentro de una carpeta
def video_name_getter(Event):
    path = os.path.join(data_folder, "Anomaly-Videos", Event)
    print(f"Buscando videos en: {path}")

    if not os.path.exists(path):
        print(f"Carpeta no encontrada: {path}")
        return []

    names = os.listdir(path)
    if ".DS_Store" in names:
        names.remove(".DS_Store")
    # Si empiezan con ._ eliminarlos
    names = [name for name in names if not name.startswith("._")]
    names.sort()
    return names

# Guardar todos los eventos en un archivo CSV
def save_events(events):
    folder = os.path.join(os.path.dirname(__file__),"../data/raw")
    archivo = "eventos.csv"
    with open(os.path.join(folder, archivo), "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(events)  # Guardar múltiples eventos correctamente

def get_csv_events():
    try:
        folder = os.path.join(os.path.dirname(__file__),"../data/raw")
        print(f"Buscando archivo CSV en: {folder}")
        archivo = "eventos.csv"
        with open(os.path.join(folder, archivo), "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            print("Eventos cargados desde el archivo CSV")
            return list(reader)
    except FileNotFoundError:
        print("Archivo CSV no encontrado.")
        exit()


def process_video(video_name, event):
    video_path = os.path.join(data_folder, "Anomaly-Videos", event, video_name)  # Ruta completa del video
    print(f"Procesando: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return []  # Retornar lista vacía si el video no se puede abrir

    frame_count = 0
    events = []  # Lista para guardar eventos de este video
    pcb_frame = None  # Almacena el frame de PCB
    start_frame = None
    end_frame = None
    registered = False

    def save_event(pcb, start, end):
        """Guarda el evento en la lista de eventos del video actual"""
        nonlocal registered
        if start is not None and end is not None:
            events.append([video_name, event, pcb if pcb is not None else None, start, end])
            print(f"Evento guardado: PCB={pcb}, Start={start}, End={end}")
            registered = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Salir cuando se acabe el video

        frame_count += 1
        cv2.imshow('Video', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):  # Marcar PCB (Previous Behaviour Crime)
            pcb_frame = frame_count
            print(f"PCB registrado en el frame: {pcb_frame}")
        elif key == ord('s'):  # Marcar inicio del evento
            start_frame = frame_count
            print(f"Inicio del evento en el frame: {start_frame}")
        elif key == ord('e'):  # Marcar fin del evento
            if start_frame is not None:
                end_frame = frame_count
                save_event(pcb_frame, start_frame, end_frame)
                start_frame = None
                end_frame = None
                pcb_frame = None

        elif key == ord('q'):  # Salir del programa
            break

    if start_frame is not None and end_frame is None:
        end_frame = frame_count
        save_event(pcb_frame, start_frame, end_frame)
        start_frame = None
        end_frame = None
        pcb_frame = None

    cap.release()
    cv2.destroyAllWindows()

    if not registered:
        print("No se registraron eventos en este video.")
        events.append([video_name, event, None, None, None, -1, -1])

    confirmation = input("¿Desea repetir? (s/n): ")
    if confirmation.lower() == "s":
        return process_video(video_name, event)  # Volver a procesar el mismo video

    return events  # Retornar eventos capturados

# main
def main():
    # Cargar eventos ya procesados
    csv_events = get_csv_events()
    print(csv_events)
    # Iterar sobre cada tipo de evento
    for event in Events:
        print(f"Procesando evento: {event}")
        videos = video_name_getter(event)

        # Mostrar los primeros 5 videos encontrados
        print(videos[:5])

        for video_name in videos:
            if video_name in [row[0] for row in csv_events]:  # Verificar si ya fue procesado
                print(f"El video {video_name} ya fue procesado. Saltando...")
                continue

            events = process_video(video_name, event)
            if events:
                save_events(events)

if __name__ == "__main__":
    main()