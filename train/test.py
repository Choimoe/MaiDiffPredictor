import json
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from tools.inference.src.enhance_json_with_features import process_json_file
from tools.inference.src.inference import predict_difficulty
from draw.visualize_diff import visualize_difficulty

song_id = "689"
diff_id = "4"

diff_name_mapping = {
    "1": "Basic",
    "2": "Advanced",
    "3": "Expert",
    "4": "Master",
    "5": "Re:Master"
}

json_file_path = f"serialized_data/{song_id}_{diff_id}.json"


def get_song_info(song_id, songs_file="info/songs.json"):
    with open(songs_file, "r", encoding="utf-8") as f:
        songs = json.load(f)
        for song in songs:
            if str(song["id"]) == str(song_id):
                return song["name"]
    return "Unknown Song"


def main():
    process_json_file(json_file_path, "chart.json")

    times, difficulty_curve = predict_difficulty(json_file_path)
    # print(difficulty_curve)
    song_name = get_song_info(song_id)
    diff_name = diff_name_mapping.get(diff_id, "Unknown")
    visualize_difficulty(times, difficulty_curve, song_name, diff_name)


if __name__ == "__main__":
    main()
