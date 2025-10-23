import pickle

import pandas as pd

from TrainingTrackNet import BallTrackerNet
import torch
import cv2
from TrainingTrackNet import postprocess
from tqdm import tqdm
import numpy as np
from utils import (
    get_bounding_box_from_center,
    get_bounding_boxes_array
)
import argparse
from itertools import groupby
from scipy.spatial import distance
# importing timer from time module
from timeit import default_timer as timer


class BallTrackerTracknet:
    def __init__(self, model_path):
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    def detect_frames(self, frames,read_from_stub=False, stub_path_ball_track=None, stub_path_dists=None):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
            model: pretrained model
        :return
            ball_track: list of detected ball points
            dists: list of euclidean distances between two neighbouring ball points
        """
        timer_start = timer()
        if read_from_stub and stub_path_ball_track is not None:
            with open(stub_path_ball_track, 'rb') as f:
                ball_track = pickle.load(f)
            with open(stub_path_dists, 'rb') as f:
                dists = pickle.load(f)
            return ball_track, dists

        device = torch.device('cpu')
        height = 360
        width = 640
        dists = [-1] * 2
        ball_track = [(None, None)] * 2
        for num in tqdm(range(2, len(frames))):
            img = cv2.resize(frames[num], (width, height))
            img_prev = cv2.resize(frames[num - 1], (width, height))
            img_preprev = cv2.resize(frames[num - 2], (width, height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = postprocess(output)
            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] and ball_track[-2][0]:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)
            if stub_path_ball_track is not None:
                with open(stub_path_ball_track, 'wb') as f:
                    pickle.dump(ball_track, f)
            if stub_path_dists is not None:
                with open(stub_path_dists, 'wb') as f:
                    pickle.dump(dists, f)
        print("Time taken to detect ball in all frames: ", timer()-timer_start, " seconds")
        return ball_track, dists

    def remove_outliers(self,ball_track, dists, max_dist=100):
        """ Remove outliers from model prediction
        :params
            ball_track: list of detected ball points
            dists: list of euclidean distances between two neighbouring ball points
            max_dist: maximum distance between two neighbouring ball points
        :return
            ball_track: list of ball points
        """
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
                ball_track[i] = (None, None)
                outliers.remove(i)
            elif dists[i - 1] == -1:
                ball_track[i - 1] = (None, None)
        return ball_track

    def draw_bounding_box_frame(self, image, keypoints):
        """
        Dessine les bounding boxes autour des keypoints fournis.

        Args:
            image (numpy.ndarray): l'image sur laquelle dessiner les bounding boxes.
            keypoints (tuple): un tuple contenant les coordonnées (x, y) du centre de la balle.
        """

        x1, y1, x2, y2 = keypoints

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        if x1 == 0 or y1==0 is None:
            return image

        # Dessine la bounding box sur l'image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        return image

    def draw_bounding_boxes_frames(self, video_frames, ball_detections):
        """
            Dessine les bounding boxes autour des keypoints de chaque frame de la vidéo.

            Args:
                video_frames (list): la liste des frames de la vidéo.
                ball_detections (list): la liste des tuples contenant les coordonnées (x, y) du centre de la balle pour chaque frame.

            Returns:
                list: la liste des frames avec les bounding boxes dessinées.
            """
        processed_frames = []

        for i, frame in enumerate(video_frames):
            # Récupère les coordonnées du keypoint pour la frame courante
            keypoint = ball_detections[i]
            print(keypoint)
            # Dessine la bounding box sur la frame
            frame_with_bbox = self.draw_bounding_box_frame(frame, keypoint)

            processed_frames.append(frame_with_bbox)

        return processed_frames

    def interpolate_ball_positons_tracknet(self, ball_positions):

        ball_positions = get_bounding_boxes_array(ball_positions)

        #convert the list into a pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        #interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() # permet de gérer le problème si la première frame n'a pas de valeur

        ball_positions = [x for x in df_ball_positions.to_numpy().tolist()] # créée une liste de dictionnaires ou 1 c'est le track id et x la bounding box

        return ball_positions
