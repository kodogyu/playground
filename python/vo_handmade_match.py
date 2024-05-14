import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import csv
import numpy as np


MOUSE_POINT = ()  # int tuple (x, y)
CORNER_POINT = ()  # float tuple (x, y)
CURRENT_MAX_FEATURE_IDX = 0
STARTING_FEATURE_IDX = 0
FEATURE_WINDOW_NAME = "Feature window"
FEATURE_IMAGE_PATCH_WINDOW_NAME = "image patch"
PREV_IMAGE_WINDOW_NAME = "previous image"
PREV_IMAGE_PATCH_WINDOW_NAME = "previous image lens"
IMAGE_SCALE = 1
GOAL_FEATURE_COUNT = 20

# function to display the image patch
# around the points clicked on the image
def click_event_feature_win(event, x, y, flags, prev_image_gray):
    global MOUSE_POINT
    global CORNER_POINT

    # checking for left mouse clicks
    if event == cv2.EVENT_MOUSEMOVE:
        x, y = x//IMAGE_SCALE, y//IMAGE_SCALE

        # set mouse point
        MOUSE_POINT = (x, y)

    # get subpixel
    elif event == cv2.EVENT_LBUTTONDOWN:
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)

        corner = np.array(MOUSE_POINT, ndmin=3, dtype=np.float32)
        corner = cv2.cornerSubPix(prev_image_gray, corner, winSize, zeroZone, criteria)

        print(f"refined corner: {corner}")
        # set mouse point
        MOUSE_POINT = (round(corner[0, 0, 0]), round(corner[0, 0, 1]))
        CORNER_POINT = (corner[0, 0, 0], corner[0, 0, 1])

    # display the image patch
    drawImagePatch(FEATURE_IMAGE_PATCH_WINDOW_NAME, prev_image, MOUSE_POINT[0], MOUSE_POINT[1])

def click_event_prev_win(event, x, y, flags, prev_image):
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = x//IMAGE_SCALE, y//IMAGE_SCALE
        # display the image patch
        drawImagePatch(PREV_IMAGE_PATCH_WINDOW_NAME, prev_image, x, y)

def drawImagePatch(window_name, image, center_x, center_y):
    patch_size = 30
    resized_patch_size = 300

    print(f"Patch center: {center_x}, {center_y}")

    image_patch = image[center_y - patch_size//2 : center_y + patch_size//2 + 1, center_x - patch_size//2 : center_x + patch_size//2 + 1]
    # resized_image_patch = cv2.resize(image_patch, (resized_patch_size, resized_patch_size), None)
    resized_image_patch = cv2.resize(image_patch, (resized_patch_size, resized_patch_size), cv2.INTER_NEAREST_EXACT)

    cv2.line(resized_image_patch, (resized_patch_size//2, 0), (resized_patch_size//2, resized_patch_size), (0, 0, 255), 2)
    cv2.line(resized_image_patch, (0, resized_patch_size//2), (resized_patch_size, resized_patch_size//2), (0, 0, 255), 2)

    cv2.imshow(window_name, resized_image_patch)

def runOnce(prev_image_idx, prev_image):
    # convert color
    prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    # curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

    # # feature extraction
    # orb = cv2.ORB_create(3000, 1.2, 8, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31, 25)

    # prev_kp, prev_des = orb.detectAndCompute(prev_image_gray, None)
    # curr_kp1, curr_des = orb.detectAndCompute(curr_image_gray, None)

    # # feature matching
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(prev_des, curr_des)
    # matches = sorted(matches, key=lambda x: x.distance)

    # # mark matches
    # for i in matches[:30]:
    #     idx = i.queryIdx
    #     x1, y1 = prev_kp[idx].pt
    #     cv2.circle(prev_image, (int(x1), int(y1)), 3, (0, 255, 0), 1)
    cv2.putText(prev_image, f"frame {prev_image_idx}", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    # scaled_prev_image = cv2.resize(prev_image, (prev_image.shape[1] * IMAGE_SCALE, prev_image.shape[0] * IMAGE_SCALE), None)
    scaled_prev_image = cv2.resize(prev_image, (prev_image.shape[1] * IMAGE_SCALE, prev_image.shape[0] * IMAGE_SCALE), cv2.INTER_NEAREST_EXACT)
    cv2.imshow(FEATURE_WINDOW_NAME, scaled_prev_image)

    global MOUSE_POINT
    global CORNER_POINT

    cv2.setMouseCallback(FEATURE_WINDOW_NAME, click_event_feature_win, prev_image_gray)

    global CURRENT_MAX_FEATURE_IDX, STARTING_FEATURE_IDX, GOAL_FEATURE_COUNT
    feature_idx = STARTING_FEATURE_IDX
    feature_dict = {}
    input_key = 0
    # 'enter'   13
    # 'space'   32
    # 'esc'     27
    # 'f'       102
    # 'r'       114
    # 's'       115
    # '왼쪽'    81
    # '위'      82
    # '오른쪽'  83
    # '아래'    84
    while len(feature_dict) < GOAL_FEATURE_COUNT:
        input_key = cv2.waitKey(0)
        print(f"Pressed [{input_key}, '{chr(input_key)}']")

        if input_key == 13:  # 'ENTER'
            if feature_idx in feature_dict:  # 추가하는 feature가 이미 있는 경우
                print(f"feature_idx [{feature_idx}] already exists. Origianlly {feature_dict[feature_idx]}. Overwriting.")
            else:
                CURRENT_MAX_FEATURE_IDX += 1

            print(f"saved feature point: {CORNER_POINT}")
            feature_dict[feature_idx] = CORNER_POINT
            print(f"[{feature_idx}] [feature count / goal feature count] : [{len(feature_dict)}/{GOAL_FEATURE_COUNT}]")

            feature_idx += 1
            updateFeatureWindow(prev_image, feature_dict)

        elif input_key == 27:  # 'ESC'
            print("Exiting...")
            break

        elif input_key == 81:  # 왼쪽
            MOUSE_POINT = (MOUSE_POINT[0] - 1, MOUSE_POINT[1])
            drawImagePatch(FEATURE_IMAGE_PATCH_WINDOW_NAME, prev_image, MOUSE_POINT[0], MOUSE_POINT[1])

        elif input_key == 82:  # 위
            MOUSE_POINT = (MOUSE_POINT[0], MOUSE_POINT[1] - 1)
            drawImagePatch(FEATURE_IMAGE_PATCH_WINDOW_NAME, prev_image, MOUSE_POINT[0], MOUSE_POINT[1])

        elif input_key == 83:  # 오른쪽
            MOUSE_POINT = (MOUSE_POINT[0] + 1, MOUSE_POINT[1])
            drawImagePatch(FEATURE_IMAGE_PATCH_WINDOW_NAME, prev_image, MOUSE_POINT[0], MOUSE_POINT[1])

        elif input_key == 84:  # 아래
            MOUSE_POINT = (MOUSE_POINT[0], MOUSE_POINT[1] + 1)
            drawImagePatch(FEATURE_IMAGE_PATCH_WINDOW_NAME, prev_image, MOUSE_POINT[0], MOUSE_POINT[1])

        elif input_key == 102:  # 'f'
            feature_num = input("feature index:")
            try :
                feature_idx = int(feature_num)
            except ValueError:
                print("feature index must be an integer.")

        elif input_key == 115:  # 's'
            starting_feature_num = input("starting feature index:")
            try :
                starting_feature_num_i = int(starting_feature_num)
                if starting_feature_num_i > 0:
                    STARTING_FEATURE_IDX = starting_feature_num_i
                else:
                    print("starting feature index must be greater than 0.")
            except ValueError:
                print("starting feature index must be an integer.")
            finally:
                feature_idx = STARTING_FEATURE_IDX

    # cv2.destroyAllWindows()
    return feature_dict

def updateFeatureWindow(image, feature_dict):
    window_image = deepcopy(image)
    for item in feature_dict.items():
        feature_pt_x, feature_pt_y = round(item[1][0]), round(item[1][1])
        cv2.rectangle(window_image, (feature_pt_x - 5, feature_pt_y - 5), (feature_pt_x + 5, feature_pt_y + 5), (255, 0, 0), 1)
        cv2.putText(window_image, f"{item[0]}", tuple(map(int, item[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

    # scaled_window_image = cv2.resize(window_image, (image.shape[1] * IMAGE_SCALE, image.shape[0] * IMAGE_SCALE), None)
    scaled_window_image = cv2.resize(window_image, (image.shape[1] * IMAGE_SCALE, image.shape[0] * IMAGE_SCALE), cv2.INTER_NEAREST_EXACT)
    cv2.imshow(FEATURE_WINDOW_NAME, scaled_window_image)

def displayPrevImage(frame_feature_dict, prev_image):
    for feature_idx in frame_feature_dict.keys():
        feature_point = tuple(map(round, frame_feature_dict[feature_idx]))
        prev_image = cv2.rectangle(prev_image, feature_point, feature_point, (0, 255, 0), 1)
        prev_image = cv2.rectangle(prev_image, (feature_point[0] - 5, feature_point[1] - 5), (feature_point[0] + 5, feature_point[1] + 5), (0, 255, 0), 1)
        prev_image = cv2.putText(prev_image, f"{feature_idx}", feature_point, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

    # scaled_prev_image = cv2.resize(prev_image, (prev_image.shape[1] * IMAGE_SCALE, prev_image.shape[0] * IMAGE_SCALE), None)
    scaled_prev_image = cv2.resize(prev_image, (prev_image.shape[1] * IMAGE_SCALE, prev_image.shape[0] * IMAGE_SCALE), cv2.INTER_NEAREST_EXACT)
    cv2.imshow(PREV_IMAGE_WINDOW_NAME, scaled_prev_image)

    cv2.setMouseCallback(PREV_IMAGE_WINDOW_NAME, click_event_prev_win, prev_image)

if __name__ == "__main__":
    # file paths
    file_list = []
    image_dir = "/home/kodogyu/Datasets/KITTI/dataset/sequences/00/image_0/"
    # image_dir = "/home/kodogyu/shared_folder_local/kitti_100_frames/"
    # image_dir = "/home/kodogyu/Datasets/TUM/rgbd_dataset_freiburg3_checkerboard_large/rgb/"

    total_frames = 3
    start_frame = 0
    for frame_idx in range(start_frame, start_frame + total_frames):
        file = image_dir + f'{frame_idx:06}.png'
        file_list.append(file)

    # tum_img0 = "1341835195.445969.png"
    # tum_img1 = "1341835195.477897.png"
    # file_list.append(image_dir + tum_img0)
    # file_list.append(image_dir + tum_img1)

    # write feature information
    fieldnames = [i for i in range(500)]
    feature_file = "files/frames_feature_info.csv"
    file = open(feature_file, 'w')
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # get GT feature matches
    frame_feature_dict_list = []
    for i in range(total_frames):
        prev_image = cv2.imread(file_list[i])

        frame_feature_dict = runOnce(i, prev_image)
        if (frame_feature_dict == {}):
            break

        print(frame_feature_dict)
        frame_feature_dict_list.append(frame_feature_dict)
        writer.writerow(frame_feature_dict)

        displayPrevImage(frame_feature_dict, prev_image)

    file.close()
