from copy import deepcopy

import cv2
import pandas as pd
from utils import (read_video, save_video, measure_distance, convert_pixel_distance_to_meters, player_stats_drawer_utils)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from trackers import BallTrackerTracknet
import constants

def main():
    #read video
    input_video_path = "input_videos/Clip1Taipei.mp4"
    video_frames = read_video(input_video_path)

    #detect players with yolo
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')


    #Detect ball with yolo
    ball_tracker = BallTracker(model_path='models/yolo_best_tennis_dataset_new.pt')

    #ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False,stub_path='tracker_stubs/ball_detections.pkl')
    #ball_detections = ball_tracker.interpolate_ball_positons(ball_detections)


    #Detect ball with TrackNet
    ball_tracker_tracknet = BallTrackerTracknet(model_path='models/model_best_TrackNet.pt')
    ball_detections_tracknet, distances_tracknet = ball_tracker_tracknet.detect_frames(video_frames, read_from_stub=False, stub_path_ball_track='tracker_stubs/ball_detections_tracknet.pkl', stub_path_dists='tracker_stubs/distances_tracknet.pkl')
    ball_detections_tracknet = ball_tracker_tracknet.remove_outliers(ball_detections_tracknet, distances_tracknet)
    ball_detections_tracknet = ball_tracker_tracknet.interpolate_ball_positons_tracknet(ball_detections_tracknet)
    print("Ball detections TrackNet")
    print(ball_detections_tracknet)
    print(len(ball_detections_tracknet))



    #court line detection
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[6])


    # Filtering the players the closest to the court
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)


    #Create the mini court
    mini_court = MiniCourt(video_frames[0])

    #detect ball hits with Yolo
    #ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections, yolo=True)

    #detect ball hits with TrackNet
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections_tracknet, yolo=False)


    #convert positions to mini court positions with yolo
    #player_minicourt_detections, ball_minicourt_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints, yolo=True)

    #convert positions to mini court positions with TrackNet
    player_minicourt_detections, ball_minicourt_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections_tracknet, court_keypoints, yolo=False)


    #Player stats calculus

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 30  # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_minicourt_detections[start_frame][1],
                                                           ball_minicourt_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_minicourt_detections[start_frame]
        player_shot_ball = min(player_positions.keys(),
                               key=lambda player_id: measure_distance(player_positions[player_id],
                                                                      ball_minicourt_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_minicourt_detections[start_frame][opponent_player_id],
            player_minicourt_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court()
                                                                               )

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / \
                                                          player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / \
                                                          player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / \
                                                            player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / \
                                                            player_stats_data_df['player_1_number_of_shots']



    #Draw the bounding boxes around the players with yolo
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)

    #Draw the bounding boxes around the ball with yolo
    #output_video_frames = ball_tracker.draw_bounding_boxes(output_video_frames, ball_detections)

    #Draw the bounding boxes around the ball with TrackNet
    output_video_frames = ball_tracker_tracknet.draw_bounding_boxes_frames(output_video_frames, ball_detections_tracknet)

    #Draw the court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    #Draw the mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    #Draw the mini court positions of the players
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_minicourt_detections)
    #Draw the mini court positions of the ball
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_minicourt_detections, color=(0, 255, 255))

    #Draw the players stats
    output_video_frames = player_stats_drawer_utils.draw_player_stats(output_video_frames, player_stats_data_df)

    #Draw the frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    save_video(output_video_frames, "output_videos/VideoShort.avi")


if __name__ == "__main__":
    main()