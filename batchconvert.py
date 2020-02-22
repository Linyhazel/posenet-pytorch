import cv2
import time
import argparse
import os
import torch
import csv
import numpy
import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--video_dir', type=str, default='./videos')
parser.add_argument('--output_dir', type=str, default='./output_csv')
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ###
    model = model.to(device) ###
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    # filenames return a array of the names of the files inside the directory
    # please make sure the file is video and only one person is in the video
    filenames = [f.path for f in os.scandir(args.video_dir) if f.is_file()] 

    for f in filenames:
        csv_path = os.path.join(args.output_dir, os.path.relpath(f, args.video_dir))
        csv_path = csv_path[0:csv_path.rfind('.')]
        csv_path += ".csv"
        #csv_content = [["nose.x", "nose.y", "leftEye.x", "leftEye.y", "rightEye.x", "rightEye.y", "leftEar.x", "leftEar.y", "rightEar.x", "rightEar.y", "leftShoulder.x", "leftShoulder.y", "rightShoulder.x", "rightShoulder.y", "leftElbow.x", "leftElbow.y", "rightElbow.x", "rightElbow.y", "leftWrist.x", "leftWrist.y", "rightWrist.x", "rightWrist.y", "leftHip.x", "leftHip.y", "rightHip.x", "rightHip.y", "leftKnee.x", "leftKnee.y", "rightKnee.x", "rightKnee.y", "leftAnkle.x", "leftAnkle.y", "rightAnkle.x", "rightAnkle.y"]]
        csv_content = [["nose.x", "leftEye.x", "rightEye.x", "leftEar.x", "rightEar.x", "leftShoulder.x", "rightShoulder.x", "leftElbow.x", "rightElbow.x", "leftWrist.x", "rightWrist.x", "leftHip.x", "rightHip.x", "leftKnee.x", "rightKnee.x", "leftAnkle.x", "rightAnkle.x", "nose.y", "leftEye.y", "rightEye.y", "leftEar.y", "rightEar.y", "leftShoulder.y", "rightShoulder.y", "leftElbow.y", "rightElbow.y", "leftWrist.y", "rightWrist.y", "leftHip.y", "rightHip.y", "leftKnee.y", "rightKnee.y", "leftAnkle.y", "rightAnkle.y"]]
        csv_content_coordinates = []
        csv_content_coordinates_x = []
        csv_content_coordinates_y = []

        cap = cv2.VideoCapture(f)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #print('Total number of frames in this video is #%d' % frame_count)
        no_of_frame = 0

        right_most_x = 0
        left_most_x = width
        uppest_y = height
        lowest_y = 0

        while no_of_frame < frame_count:
            no_of_frame += 1 # haven't implement pick frame, read frame one by one
            input_image, display_image, output_scale = posenet.read_cap(cap, scale_factor=args.scale_factor, output_stride=output_stride)
            if input_image is None:
                break

            with torch.no_grad():
                input_image = torch.Tensor(input_image).to(device) ###

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

            keypoint_coords *= output_scale

            if not args.notxt:
                #print("Frame No.%s" % no_of_frame)
                #print("Results for video: %s" % f)
                for pi in range(len(pose_scores)): # there should be one pi only -> one person in the video
                    if pose_scores[pi] == 0.: # drop frames that movements cannot be recognized
                        break
                    #print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    #row = []
                    row_x = []
                    row_y = []
                    #for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                     #   print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

                    for i in range(17):
                        x_coordinate = float(keypoint_coords[pi][i][0])
                        y_coordinate = float(keypoint_coords[pi][i][1])
                        if(x_coordinate < left_most_x):
                            left_most_x = x_coordinate
                        if(x_coordinate > right_most_x):
                            right_most_x = x_coordinate
                        if(y_coordinate > lowest_y):
                            lowest_y = y_coordinate
                        if(y_coordinate < uppest_y):
                            uppest_y = y_coordinate

                        row_x.append(float(keypoint_coords[pi][i][0]))
                        row_y.append(float(keypoint_coords[pi][i][1]))
                        #row.append(keypoint_coords[pi][i][0])

                    csv_content_coordinates_x.append(row_x)
                    csv_content_coordinates_y.append(row_y)
                    #csv_content.append(row)
        
        csv_content_coordinates_x = numpy.array(csv_content_coordinates_x)
        csv_content_coordinates_y = numpy.array(csv_content_coordinates_y)
        csv_content_coordinates_x = csv_content_coordinates_x - left_most_x
        csv_content_coordinates_y = csv_content_coordinates_y - uppest_y 
        body_width = right_most_x - left_most_x
        body_height = lowest_y - uppest_y
        csv_content_coordinates_x = csv_content_coordinates_x/body_width
        csv_content_coordinates_y = csv_content_coordinates_y/body_height
        csv_content_coordinates = numpy.append(csv_content_coordinates_x, csv_content_coordinates_y , axis=1)
        print("shape of the %s content is: " % f)
        print(csv_content_coordinates.shape)
        #csv_content.append(csv_content_coordinates.tolist())

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_content)
            writer.writerows(csv_content_coordinates)


if __name__ == "__main__":
    main()
