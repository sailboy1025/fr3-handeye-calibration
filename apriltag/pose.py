import cv2
import numpy as np
from pupil_apriltags import Detector

# ====== 你需要改的：tag真实边长（单位：米）======
TAG_SIZE_M = 0.10  # 例如 4 cm 的 tag 就写 0.04

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=.1,
    quad_sigma=0.0,
    refine_edges=1
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera (check macOS camera permission).")

def get_camera_params_from_frame(frame):
    """
    你没有标定文件时的“临时近似”：
    fx≈fy≈max(w,h)，cx=w/2，cy=h/2
    能跑，但尺度/姿态会有系统误差。想准必须做标定。
    """
    h, w = frame.shape[:2]
    f = max(w, h) * 1.2
    return (f, f, w / 2.0, h / 2.0)

def draw_axes(frame, K, rvec, tvec, axis_len=0.03):
    # 画 3D 坐标轴（单位：米）
    pts_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ])
    dist = np.zeros((4, 1))
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    p0, px, py, pz = pts_2d.reshape(-1, 2).astype(int)

    cv2.line(frame, tuple(p0), tuple(px), (0, 0, 255), 3)   # X 红
    cv2.line(frame, tuple(p0), tuple(py), (0, 255, 0), 3)   # Y 绿
    cv2.line(frame, tuple(p0), tuple(pz), (255, 0, 0), 3)   # Z 蓝

def draw_cube(frame, K, rvec, tvec, cube_size=0.05):
    """Draw a 3D cube based on pose"""
    cube_pts_3d = np.float32([
        [0, 0, 0], [cube_size, 0, 0], [cube_size, cube_size, 0], [0, cube_size, 0],
        [0, 0, cube_size], [cube_size, 0, cube_size], [cube_size, cube_size, cube_size], [0, cube_size, cube_size]
    ])
    dist = np.zeros((4, 1))
    cube_pts_2d, _ = cv2.projectPoints(cube_pts_3d, rvec, tvec, K, dist)
    cube_pts_2d = cube_pts_2d.reshape(-1, 2).astype(int)
    
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    for edge in edges:
        cv2.line(frame, tuple(cube_pts_2d[edge[0]]), tuple(cube_pts_2d[edge[1]]), (200, 200, 0), 2)
while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fx, fy, cx, cy = get_camera_params_from_frame(frame)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=TAG_SIZE_M
    )

    for d in detections:
        # 画框
        corners = d.corners.astype(int)
        for i in range(4):
            cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 0, 255), 2)

        # pupil-apriltags 给 pose_R (3x3) 和 pose_t (3x1)
        R = d.pose_R
        t = d.pose_t  # meters, shape (3,1)

        # 转成 OpenCV 的 rvec/tvec 以便 projectPoints
        # rvec, _ = cv2.Rodrigues(R)
        # tvec = t.reshape(3, 1)
        R_fix = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
        R2 = R @ R_fix

        rvec, _ = cv2.Rodrigues(R2)
        tvec = t.reshape(3, 1)


        draw_axes(frame, K, rvec, tvec, axis_len=0.03)
        draw_cube(frame, K, rvec, tvec, cube_size=0.03)


        # 显示平移（米）
        # tx, ty, tz = tvec.ravel()
        # cv2.putText(frame, f"id={d.tag_id} t=[{tx:.3f},{ty:.3f},{tz:.3f}] m",
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("AprilTag Pose (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()