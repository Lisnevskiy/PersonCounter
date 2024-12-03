import json

import pytest
from shapely.geometry import LineString

from main import (
    filter_duplicate_actions,
    get_configuration,
    people_count,
    process_frames,
    read_json,
    scale_to_line,
    update_visitor_status,
)


@pytest.fixture
def sample_json_data():
    return {
        "eventSpecific": {
            "nnDetect": {
                "10_8_3_203_rtsp_camera_3": {
                    "cfg": {
                        "cross_lines": [
                            {
                                "ext_line": [200, 100, 300, 100],
                                "int_line": [150, 200, 250, 200],
                                "box": [836, 470],
                            }
                        ],
                        "video_frames": {"frame_width": 640, "frame_height": 360, "media_fn": 159},
                    },
                    "frames": {
                        "1698080399016990": {
                            "timestamp": 1698080399.01699,
                            "detected": {
                                "person": [[140, 190, 260, 210, 0.95, {"1698069301:27": {"track_id": "1698069301:27"}}]]
                            },
                        },
                        "1698080400000000": {
                            "timestamp": 1698080400,
                            "detected": {
                                "person": [[240, 90, 260, 110, 0.88, {"1698069301:27": {"track_id": "1698069301:27"}}]]
                            },
                        },
                        "1698080410000000": {
                            "timestamp": 1698080410,
                            "detected": {
                                "person": [[240, 190, 260, 210, 0.92, {"1698069301:28": {"track_id": "1698069301:28"}}]]
                            },
                        },
                    },
                }
            }
        }
    }


def test_read_json(tmp_path):
    """Тестирование функции чтения JSON."""
    test_file = tmp_path / "test.json"
    test_data = {"key": "value"}
    with open(test_file, "w") as f:
        json.dump(test_data, f)

    result = read_json(str(test_file))
    assert result == test_data, "Данные из файла должны быть корректно прочитаны"


def test_read_json_error(tmp_path):
    """Тестирование обработки ошибок при чтении JSON."""
    non_existent_file = tmp_path / "non_existent.json"
    result = read_json(str(non_existent_file))
    assert result == {}, "Результат должен быть пустым словарем для несуществующего файла"


def test_get_configuration(sample_json_data):
    """Тестирование извлечения конфигурации."""
    int_line, ext_line, frames = get_configuration(sample_json_data)

    assert isinstance(int_line, LineString), "Входная линия должна быть LineString"
    assert isinstance(ext_line, LineString), "Выходная линия должна быть LineString"
    assert not int_line.intersects(ext_line), "Линии не должны пересекаться"


def test_scale_to_line():
    """
    Тестирование функции scale_to_line.
    Проверяет корректное масштабирование координат и создание LineString.
    """
    coords = [100, 200, 300, 400]
    box_dimensions = (1000, 1000)
    frame_dimensions = (500, 500)

    expected_line = LineString([(50.0, 100.0), (150.0, 200.0)])
    result_line = scale_to_line(coords, box_dimensions, frame_dimensions)

    assert isinstance(result_line, LineString), "Результат должен быть объектом LineString"
    assert list(result_line.coords) == list(expected_line.coords), "Координаты должны быть корректно масштабированы"


def test_update_visitor_status_with_entry_and_exit():
    """Тестируем последовательность входа и выхода."""
    diagonal_entry1 = LineString([(140, 190), (260, 210)])
    diagonal_exit1 = LineString([(240, 90), (260, 110)])
    diagonal_entry2 = LineString([(260, 210), (140, 190)])
    diagonal_exit2 = LineString([(260, 110), (240, 90)])
    int_line = LineString([(150, 200), (250, 200)])
    ext_line = LineString([(200, 100), (300, 100)])
    visitors = {}
    track_id = "track1"
    timestamp = 1698080399

    # Вход
    visitors = update_visitor_status(
        diagonal_entry1, diagonal_entry2, int_line, ext_line, track_id, timestamp, visitors
    )

    # Выход
    timestamp += 1
    visitors = update_visitor_status(diagonal_exit1, diagonal_exit2, int_line, ext_line, track_id, timestamp, visitors)
    assert visitors[track_id]["actions"][-1] == {"timestamp": timestamp, "action": "EXT"}

    # Выход
    track_id = "track2"
    visitors = update_visitor_status(diagonal_exit1, diagonal_exit2, int_line, ext_line, track_id, timestamp, visitors)

    # Вход
    timestamp += 1
    visitors = update_visitor_status(
        diagonal_entry1, diagonal_entry2, int_line, ext_line, track_id, timestamp, visitors
    )
    assert visitors[track_id]["actions"] == [{"timestamp": timestamp, "action": "INT"}]


def test_filter_duplicate_actions():
    """Тестирование фильтрации дублирующихся действий."""
    actions = [
        {"timestamp": 1, "action": "INT"},
        {"timestamp": 2, "action": "INT"},
        {"timestamp": 3, "action": "EXT"},
        {"timestamp": 4, "action": "EXT"},
        {"timestamp": 5, "action": "INT"},
    ]

    filtered_actions = filter_duplicate_actions(actions)

    assert len(filtered_actions) == 2, "Должно остаться 2 уникальных действия"
    assert [action for action in filtered_actions] == [
        "INT",
        "EXT",
    ], "Порядок действий должен быть сохранен"


def test_people_count(sample_json_data):
    """Тестирование подсчета людей."""
    visitors = {
        "track1": {"actions": [{"timestamp": 1, "action": "INT"}, {"timestamp": 2, "action": "EXT"}]},
        "track2": {"actions": [{"timestamp": 3, "action": "INT"}]},
        "track3": {"actions": [{"timestamp": 5, "action": "INT"}, {"timestamp": 6, "action": "EXT"}]},
    }

    entry, exit, current = people_count(visitors)

    assert entry == 3, "Ожидается 3 вошедших"
    assert exit == 2, "Ожидается 2 вышедших"
    assert current == 1, "Ожидается 1 текущий посетитель"


def test_process_frames(sample_json_data):
    """Тестирование обработки кадров."""
    int_line, ext_line, frames = get_configuration(sample_json_data)
    visitors = {}

    result_visitors = process_frames(int_line, ext_line, frames, visitors)

    assert len(result_visitors) == 2, "Должно быть обработано два трека"
    assert "1698069301:27" in result_visitors, "Трек '1698069301:27' должен быть в результатах"
    assert result_visitors["1698069301:27"]["actions"] == [], "Ожидается []"


def test_multiple_entries_and_exits():
    """Тестирование нескольких входов и выходов."""
    visitors = {
        "track1": {"actions": [{"timestamp": 1, "action": "INT"}, {"timestamp": 2, "action": "EXT"}]},
        "track2": {"actions": [{"timestamp": 3, "action": "INT"}, {"timestamp": 4, "action": "INT"}]},
        "track3": {"actions": [{"timestamp": 5, "action": "EXT"}]},
    }

    entry, exit, current = people_count(visitors)

    assert entry == 2, "Ожидается 2 вошедших"
    assert exit == 2, "Ожидается 2 вышедших"
    assert current == 1, "Ожидается 1 текущий посетитель"


def test_repeated_exit():
    """Тестирование повторного выхода одного трека."""
    visitors = {
        "track1": {"actions": [{"timestamp": 1, "action": "EXT"}, {"timestamp": 2, "action": "EXT"}]},
    }

    entry, exit, current = people_count(visitors)

    assert entry == 0, "Не должно быть входов"
    assert exit == 1, "Должен быть учтен только один выход"
    assert current == 0, "Не должно остаться посетителей"


def test_frame_with_no_people(sample_json_data):
    """Тестирование кадра без людей."""
    sample_json_data["eventSpecific"]["nnDetect"]["10_8_3_203_rtsp_camera_3"]["frames"]["1698080410000001"] = {
        "timestamp": 1698080411,
        "detected": {"person": []},
    }

    int_line, ext_line, frames = get_configuration(sample_json_data)
    visitors = {}

    result_visitors = process_frames(int_line, ext_line, frames, visitors)

    assert len(result_visitors) == 2, "Должно остаться два трека"
