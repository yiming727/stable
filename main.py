import cv2
import numpy as np

from fixborder import fix_border, extreme_corners, min_auto_border_size
from smooth import TrajectorySmoother
from superglue import process_video_to_npz


def stabilize_video(input_path,output_path,smoothing=50,grid_x=4,grid_y=4,max_translation=15,max_rotation=15,confidence_threshold=0.75):

    SMOOTHING_RADIUS = smoothing
    GRID_SIZE = (grid_x, grid_y)
    MAX_TRANSLATION_THRESHOLD = max_translation
    MAX_ROTATION_THRESHOLD = np.deg2rad(max_rotation)
    CONFIDENCE_THRESHOLD = confidence_threshold

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    _, prev = cap.read()
    if prev is None:
        print("Error reading video file")
        cap.release()
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames - 1, 3), np.float32)
    grid_w = w // GRID_SIZE[0]
    grid_h = h // GRID_SIZE[1]

    # superglue特征匹配
    npz_files = process_video_to_npz(input_path)
    for key, file in enumerate(npz_files):
        data = np.load(file)
        keypoints0 = data['keypoints0']
        keypoints1 = data['keypoints1']
        matches = data['matches']
        match_confidence = data['match_confidence']
        matched_keypoints0 = []
        matched_keypoints1 = []
        for i, match in enumerate(matches):
            if match >= 0 and match_confidence[i] >= CONFIDENCE_THRESHOLD:
                matched_keypoints0.append(keypoints0[i])
                matched_keypoints1.append(keypoints1[match])
        matched_keypoints0 = np.array(matched_keypoints0)
        matched_keypoints1 = np.array(matched_keypoints1)
        if matched_keypoints0.shape[0] < 4:
            affine_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            affine_matrix, _ = cv2.estimateAffinePartial2D(matched_keypoints0, matched_keypoints1)
        if affine_matrix is None:
            affine_matrix = np.eye(2, 3, dtype=np.float32)
        dx = affine_matrix[0, 2]
        dy = affine_matrix[1, 2]
        da = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
        transforms[key] = [dx, dy, da]

    trajectory = np.cumsum(transforms, axis=0)
    smoother = TrajectorySmoother(smoothing_radius=SMOOTHING_RADIUS)
    smoothed_trajectory = smoother.smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    extreme_frame_corners = extreme_corners(prev_gray, transforms_smooth)
    border_size = min_auto_border_size(extreme_frame_corners)

    for i in range(n_frames - 2):
        success, frame = cap.read()
        if not success:
            break
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        new_frame = fix_border(frame_stabilized, extreme_frame_corners, border_size)
        frame_out = cv2.hconcat([frame, new_frame])
        out.write(new_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    stabilize_video('../07.mp4', '../NormEnhanced07.mp4')
