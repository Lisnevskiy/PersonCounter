import json
from typing import Any

from shapely.geometry import LineString


def read_json(file_path: str) -> dict:
    """Чтение данных из JSON файла"""

    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Ошибка при чтении JSON: {e}")
        return {}


def get_configuration(data: dict) -> tuple[LineString, LineString, dict, tuple[int, int], tuple[int, int]]:
    """Извлекает конфигурацию линий, кадров и боксов из JSON"""

    nn_detect = data["eventSpecific"]["nnDetect"]["10_8_3_203_rtsp_camera_3"]
    cfg = nn_detect["cfg"]

    lines = cfg["cross_lines"][0]
    box_dimensions = tuple(lines["box"])
    frame_dimensions = (
        cfg["video_frames"]["frame_width"],
        cfg["video_frames"].get("frame_height", 360),
    )
    frames = nn_detect["frames"]

    ext_coords = scale_coordinates(lines["ext_line"], box_dimensions, frame_dimensions)
    int_coords = scale_coordinates(lines["int_line"], box_dimensions, frame_dimensions)

    ext_line = LineString([(ext_coords[0], ext_coords[1]), (ext_coords[2], ext_coords[3])])
    int_line = LineString([(int_coords[0], int_coords[1]), (int_coords[2], int_coords[3])])

    return int_line, ext_line, frames, box_dimensions, frame_dimensions


def scale_coordinates(
    coords: list[int], box_dimensions: tuple[int, int], frame_dimensions: tuple[int, int]
) -> list[float]:
    """Масштабирует координаты с учетом размеров бокса и кадра"""

    box_width, box_height = box_dimensions
    frame_width, frame_height = frame_dimensions
    return [
        (coords[0] / box_width) * frame_width,
        (coords[1] / box_height) * frame_height,
        (coords[2] / box_width) * frame_width,
        (coords[3] / box_height) * frame_height,
    ]


def process_frames(int_line: LineString, ext_line: LineString, frames: dict, visitors: dict) -> dict:
    """Обрабатывает кадры и анализирует пересечения с линиями"""

    updated_visitors = visitors.copy()
    for frame_id, frame_data in frames.items():
        timestamp = frame_data["timestamp"]
        detected_people = frame_data["detected"].get("person", [])
        for person in detected_people:
            if len(person) > 5:
                track_id = list(person[-1].values())[0].get("track_id")
                if track_id:
                    x1, y1, x2, y2 = person[:4]
                    diagonal = LineString([(x1, y1), (x2, y2)])
                    updated_visitors = update_visitor_status(
                        diagonal, int_line, ext_line, track_id, timestamp, updated_visitors
                    )
    return updated_visitors


def update_visitor_status(
    diagonal: LineString, int_line: LineString, ext_line: LineString, track_id: str, timestamp: float, visitors: dict
) -> dict:
    """Обновляет действия посетителя: вход, выход"""

    updated_visitors = visitors.copy()
    if track_id not in updated_visitors:
        updated_visitors[track_id] = []

    # Проверка пересечения с линией входа (INT)
    if not diagonal.intersection(int_line).is_empty:
        if not any(action["action"] == "INT" for action in updated_visitors[track_id]):
            updated_visitors[track_id].append({"timestamp": timestamp, "action": "INT"})

    # Проверка пересечения с линией выхода (EXT)
    elif not diagonal.intersection(ext_line).is_empty:
        if not any(action["action"] == "EXT" for action in updated_visitors[track_id]):
            updated_visitors[track_id].append({"timestamp": timestamp, "action": "EXT"})

    return updated_visitors


def people_count(visitors: dict[str, list[dict[str, Any]]]) -> tuple[int, int, int]:
    """Считает количество вошедших, вышедших и оставшихся посетителей"""

    entry_count = 0
    exit_count = 0
    current_visitors = set()

    for track_id, actions in visitors.items():
        filtered_actions = filter_duplicate_actions(actions)
        for action in filtered_actions:
            if action["action"] == "INT" and track_id not in current_visitors:
                entry_count += 1
                current_visitors.add(track_id)
            elif action["action"] == "EXT" and track_id in current_visitors:
                exit_count += 1
                current_visitors.remove(track_id)

    return entry_count, exit_count, len(current_visitors)


def filter_duplicate_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Убирает дублирующиеся действия для одного трека"""

    filtered = []
    last_action = None
    for action in actions:
        if action["action"] != last_action:
            filtered.append(action)
            last_action = action["action"]
    return filtered


def main():
    """Основная функция запуска"""

    file_path = "detections.json"
    data = read_json(file_path)

    if not data:
        print("Ошибка: не удалось загрузить данные.")
        return

    int_line, ext_line, frames, _, _ = get_configuration(data)
    visitors = {}

    visitors = process_frames(int_line, ext_line, frames, visitors)
    entry_count, exit_count, visitors_in_shop = people_count(visitors)

    print(f"Вход: {entry_count}")
    print(f"Выход: {exit_count}")
    print(f"Людей в магазине: {visitors_in_shop}")


if __name__ == "__main__":
    main()
