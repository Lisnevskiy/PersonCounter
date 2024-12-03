import json

from shapely.geometry import LineString


def read_json(file_path: str) -> dict:
    """Чтение данных из JSON файла."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"[ERROR] Ошибка при чтении JSON: {e}")
        return {}


def get_configuration(data: dict) -> tuple[LineString, LineString, dict]:
    """
    Извлекает конфигурацию линий, кадров и боксов из JSON.

    :param data: Словарь с данными JSON.
    :return: Входная линия, выходная линия, кадры, размеры бокса, размеры кадра.
    """
    nn_detect = data["eventSpecific"]["nnDetect"]["10_8_3_203_rtsp_camera_3"]
    cfg = nn_detect["cfg"]

    lines = cfg["cross_lines"][0]
    box_dimensions = tuple(lines["box"])
    frame_dimensions = (
        cfg["video_frames"].get("frame_height", 640),
        cfg["video_frames"].get("frame_height", 360),
    )
    frames = nn_detect["frames"]

    ext_line = scale_to_line(lines["ext_line"], box_dimensions, frame_dimensions)
    int_line = scale_to_line(lines["int_line"], box_dimensions, frame_dimensions)

    return int_line, ext_line, frames


def scale_to_line(coords: list[int], box_dimensions: tuple[int, int], frame_dimensions: tuple[int, int]) -> LineString:
    """
    Масштабирует координаты и возвращает объект LineString.

    :param coords: Координаты в системе бокса.
    :param box_dimensions: Размеры бокса.
    :param frame_dimensions: Размеры кадра.
    :return: LineString в системе координат кадра.
    """
    box_width, box_height = box_dimensions
    frame_width, frame_height = frame_dimensions

    scaled = [
        (coords[0] / box_width) * frame_width,
        (coords[1] / box_height) * frame_height,
        (coords[2] / box_width) * frame_width,
        (coords[3] / box_height) * frame_height,
    ]

    return LineString([(scaled[0], scaled[1]), (scaled[2], scaled[3])])


def process_frames(int_line: LineString, ext_line: LineString, frames: dict, visitors: dict) -> dict:
    """
    Обрабатывает кадры и анализирует пересечения с линиями.

    :param int_line: Входная линия.
    :param ext_line: Выходная линия.
    :param frames: Кадры с информацией о детекции.
    :param visitors: Данные о посетителях.
    :param box_dimensions: Размеры бокса.
    :param frame_dimensions: Размеры кадра.
    :return: Обновленные данные о посетителях.
    """
    for frame_data in frames.values():
        detected_people = frame_data["detected"].get("person", [])
        for person in detected_people:
            track_id_data = person[-1]
            if isinstance(track_id_data, dict) and list(track_id_data.values())[0].get("track_id"):
                track_id = list(track_id_data.values())[0]["track_id"]
                x1, y1, x2, y2 = person[:4]

                # Углы прямоугольника
                top_left = (x1, y1)
                bottom_right = (x2, y2)
                top_right = (x2, y1)
                bottom_left = (x1, y2)

                # Построение диагоналей
                diagonal_1 = LineString([top_left, bottom_right])  # Диагональ 1
                diagonal_2 = LineString([top_right, bottom_left])  # Диагональ 2

                visitors = update_visitor_status(
                    diagonal_1, diagonal_2, int_line, ext_line, track_id, frame_data["timestamp"], visitors
                )
    return visitors


def update_visitor_status(
    diagonal_1: LineString,
    diagonal_2: LineString,
    int_line: LineString,
    ext_line: LineString,
    track_id: str,
    timestamp: float,
    visitors: dict,
) -> dict:
    """Обновляет действия посетителя: вход, выход."""
    if track_id not in visitors:
        visitors[track_id] = {"actions": [], "state": None}

    track_data = visitors[track_id]

    if diagonal_1.intersects(ext_line) or diagonal_2.intersects(ext_line):
        if track_data["state"] == "ENTRY_CROSSED":
            track_data["actions"].append({"timestamp": timestamp, "action": "EXT"})
            track_data["state"] = "EXIT_CONFIRMED"
            return visitors
        else:
            track_data["state"] = "EXIT_CROSSED"

    if diagonal_1.intersects(int_line) or diagonal_2.intersects(int_line):
        if track_data["state"] == "EXIT_CROSSED":
            track_data["actions"].append({"timestamp": timestamp, "action": "INT"})
            track_data["state"] = "ENTRY_CONFIRMED"
            return visitors
        else:
            track_data["state"] = "ENTRY_CROSSED"

    return visitors


def people_count(visitors: dict) -> tuple[int, int, int]:
    """Подсчитывает количество вошедших, вышедших и текущих посетителей."""
    entry_count = 0
    exit_count = 0
    current_visitors = set()
    processed_tracks = set()

    for track_id, data in visitors.items():
        actions = data["actions"]
        if not actions:
            continue
        filtered_actions = filter_duplicate_actions(actions)

        for action in filtered_actions:
            if action == "INT":
                if track_id not in current_visitors:
                    entry_count += 1
                    current_visitors.add(track_id)
            elif action == "EXT":
                if track_id in current_visitors:
                    exit_count += 1
                    current_visitors.remove(track_id)
                elif track_id not in processed_tracks:
                    exit_count += 1
                    processed_tracks.add(track_id)

    return entry_count, exit_count, len(current_visitors)


def filter_duplicate_actions(actions: list[dict]) -> list[str]:
    """Убирает дублирующиеся действия для одного трека."""
    filtered = []
    for action in actions:
        if action["action"] not in filtered:
            filtered.append(action["action"])
    return filtered


def main() -> None:
    file_path = "detections.json"
    data = read_json(file_path)

    if not data:
        print("Ошибка: не удалось загрузить данные.")
        return

    try:
        int_line, ext_line, frames = get_configuration(data)
    except ValueError as e:
        print(f"Ошибка конфигурации: {e}")
        return

    visitors = process_frames(int_line, ext_line, frames, {})
    entry_count, exit_count, visitors_in_shop = people_count(visitors)

    print(f"Вход: {entry_count}")
    print(f"Выход: {exit_count}")
    print(f"Людей в магазине: {visitors_in_shop}")


if __name__ == "__main__":
    main()
