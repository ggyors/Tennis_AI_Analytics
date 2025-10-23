import sys

import cv2
from ultralytics import YOLO
import pickle  # Pour sauvegarder les données dans un fichier

sys.path.append('../')
from utils import measure_distance, get_center_of_bbox


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            #avoid la personne numéro 3
            if track_id == 3:
                continue
            if track_id == 4:
                continue
            if track_id == 10:
                continue
            if track_id == 2:
                continue
            player_center = get_center_of_bbox(bbox)
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # sort distances in the ascending order
        distances.sort(key=lambda x: x[1])
        # choose the fist 2 players
        chosen_players = [distances[0][0], distances[1][0]]
        #track_id des joueurs choisis
        print(chosen_players)
        return chosen_players

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[293]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []

        """
        #old version
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections
        """

        #new version
        for player_dict in player_detections:
            new_player_dict = {}
            for track_id, bbox in player_dict.items():
                if track_id in chosen_players:
                    if track_id != 1:
                        new_player_dict[2] = bbox  # Modifie l'ID à '2'
                    else:
                        new_player_dict[track_id] = bbox
            filtered_player_detections.append(new_player_dict)
        return filtered_player_detections

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections  # renvoit la liste des dictionnaires de joueurs détectés

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]  # Persist dis que l'objet est le même entre les frames
        if results.boxes is None:
            return {}  # No players detected, return an empty dictionary

        id_name_dict = results.names
        player_dict = {}
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[
                    track_id] = result  # ajoute l'id du joueur associé aux coordonnées du joueur dans le dictionnaire
        return player_dict

    def draw_bounding_boxes(self, video_frames, player_detections):
        output_videos_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_videos_frames.append(frame)

        return output_videos_frames
