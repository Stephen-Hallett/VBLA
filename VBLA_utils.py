from collections import Counter
from itertools import combinations

import cv2 as cv
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from ultralytics import YOLO

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

player_model = YOLO("yolov10x.pt")


def match_descriptors(
    descr1: np.ndarray, descr2: np.ndarray, thresh: float = 0.7
) -> tuple[cv.DMatch]:
    results = flann.knnMatch(descr1, descr2, k=2)
    distance_mat = np.array([[n1.distance, n2.distance] for n1, n2 in results])
    match_mat = distance_mat[:, 0] < thresh * distance_mat[:, 1]
    # Bin all good matches and find the image which has the most matches
    return [result[0] for i, result in enumerate(results) if match_mat[i]]


def condensed_index(n: int, i: int, j: int) -> int:
    """Calculate the index in the condensed distance matrix for the (i, j) element."""
    if i > j:
        i, j = j, i
    return int(n * i - i * (i + 1) // 2 + j - i - 1)


def polar2cartesian(line: np.ndarray, tol: float = 1e-9) -> list[float, float]:
    if line.shape[0] == 3:
        rho, theta, _ = line
    else:
        rho, theta = line

    if theta > np.pi / 2:
        rho = -rho
        theta -= np.pi

    a = np.cos(theta)
    b = np.sin(theta)
    m = (-a) / (b + tol)  # Gradient value
    c = rho / (b + tol)  # Constant value
    return [m, c]


def gradient_from_points(point_set: np.ndarray, tol: float = 1e-9) -> float:
    run_rise = np.array([-1, 1]) @ point_set
    return run_rise[1] / (run_rise[0] + tol)


def points2cartesian(point_set: np.ndarray, tol: float = 1e-9) -> list:
    m = gradient_from_points(point_set, tol)
    c = point_set[0][1] - m * point_set[0][0]
    return [m, c]


def cartesian2points(
    line: np.ndarray, shape: tuple[list, list], tol: float = 1e-9
) -> np.array:
    # print(line)
    m = line[0]  # Gradient value
    c = line[1]  # Constant value

    # To find line intersections with screen, need to find
    # y values when x = 0 and x = shape[1] for horizontal
    # x values when y = 0 and y = shape[0] for vertical
    y_start = c
    y_end = shape[1] * m + c
    x_start = -c / (m + tol)
    x_end = (shape[0] - c) / (m + tol)
    points = []
    if sum([abs(y_start), abs(y_end)]) < sum([abs(x_start), abs(x_end)]):
        points = [[0, y_start], [shape[1], y_end]]  # Horizontal
    else:
        points = [[x_start, 0], [x_end, shape[0]]]  # Vertical

    if not points:
        points = [[np.nan, np.nan], [np.nan, np.nan]]
    return np.array(points)


def points2polar(point_set: np.ndarray, tol: float = 1e-9) -> list:
    m, c = points2cartesian(point_set, tol)
    # Return Rho, Theta
    rho = np.abs(c) / np.sqrt(m**2 + 1)
    theta = np.arctan(-1 / (m + tol))
    return [rho, theta]


def polar2points(
    line: np.ndarray, shape: tuple[list, list], tol: float = 1e-9
) -> np.array:
    if line.shape[0] == 3:
        rho, theta, _ = line
    else:
        rho, theta = line

    if theta > np.pi / 2:
        rho = -rho
        theta -= np.pi

    a = np.cos(theta)
    b = np.sin(theta)
    m = (-a) / (b + tol)  # Gradient value
    c = rho / (b + tol)  # Constant value

    # To find line intersections with screen, need to find
    # y values when x = 0 and x = shape[1] for horizontal
    # x values when y = 0 and y = shape[0] for vertical
    y_start = c
    y_end = shape[1] * m + c
    x_start = -c / (m + tol)
    x_end = (shape[0] - c) / (m + tol)
    points = []
    if sum([abs(y_start), abs(y_end)]) < sum([abs(x_start), abs(x_end)]):
        points = [[0, y_start], [shape[1], y_end]]  # Horizontal
    else:
        points = [[x_start, 0], [x_end, shape[0]]]  # Vertical

    if not points:
        points = [[np.nan, np.nan], [np.nan, np.nan]]
    return np.array(points)


def weighted_average_line(
    line: tuple[list, list, list],
) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    """Convert lists of start_points, end_points and votes into one rectanglar format line weighted by votes.

    :param line: Tuple containing lists of start_points, end_points & vote values
    :return: Weighted average start and end point values
    """
    start = np.zeros(shape=(1, 2))
    end = np.zeros(shape=(1, 2))
    total_votes = sum(line[2])
    for i in range(len(line[0])):
        weighting = line[2][i] / total_votes
        start += weighting * np.array(line[0][i])
        end += weighting * np.array(line[1][i])
    return (np.squeeze(start), np.squeeze(end)), total_votes


def scale_pointset(
    point_set: np.ndarray, scale_height: float, scale_width: float
) -> np.ndarray:
    return point_set / [[scale_width, scale_height], [scale_width, scale_height]]


def clean_lines(
    cartesian_lines: np.ndarray,
    votes: np.ndarray,
    shape: tuple[int, int],
    scale_height: float = 1,
    scale_width: float = 1,
    threshold: float = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine similar lines together based on their start, end and center points.

    :param cartesian_lines: Rows of lines in cartesian form (m, c)
    :param votes: Votes corresponding to each line
    :param shape: Shape of the image for which lines were detected
    :param scale_height: factor with which to scale the lines height, defaults to 1
    :param scale_width: factor with which to scale the lines width, defaults to 1
    :param threshold: Accepted distance to classify lines as similar, defaults to 20
    :return: Combined lines and votes
    """
    line_points = np.apply_along_axis(
        lambda row: cartesian2points(row, shape=shape), axis=1, arr=cartesian_lines
    )
    keep_indices = ~(np.isnan(line_points).any(axis=1).any(axis=1))
    votes = votes[keep_indices]
    line_points = line_points[keep_indices]

    line_starts, line_ends = line_points[:, 0, :], line_points[:, 1, :]

    line_centers = np.mean((line_starts, line_ends), axis=0)
    # Averaged lines will be a list of tuples of 3 lists
    # One tuple for each line, where each tuple contains a list for start point, end point & votes
    established_lines = []

    start_distances = pdist(line_starts, "euclidean")
    center_distances = pdist(line_centers, "euclidean")
    end_distances = pdist(line_ends, "euclidean")
    distances = (start_distances + center_distances + end_distances) / 3
    n_points = len(votes)
    unprocessed = set(range(n_points))

    for i in range(n_points):
        if i not in unprocessed:
            continue
        unprocessed.remove(i)
        line = ([line_starts[i, :]], [line_ends[i, :]], [votes[i]])
        for j in range(i + 1, n_points):
            if j not in unprocessed:
                continue
            index = condensed_index(n_points, i, j)
            if abs(distances[index]) < threshold:
                line[0].append(line_starts[j, :])
                line[1].append(line_ends[j, :])
                line[2].append(votes[j])
                unprocessed.remove(j)
            else:
                pass
        established_lines.append(line)
    averages = [weighted_average_line(line) for line in established_lines]
    averaged_lines = np.array([line for line, _ in averages])
    averaged_votes = np.array([vote for _, vote in averages])
    scaled_lines = [
        scale_pointset(points, scale_height, scale_width) for points in averaged_lines
    ]
    scaled_cartesian = np.array([points2cartesian(row) for row in scaled_lines])
    vote_ordering = averaged_votes.argsort()[::-1]

    return scaled_cartesian[vote_ordering], averaged_votes[vote_ordering]


def mask_background(image: np.ndarray) -> np.ndarray:
    lower_green = np.array(
        [40, 40, 40]
    )  # Lower bound for H (hue), L (lightness), S (saturation)
    upper_green = np.array(
        [80, 255, 255]
    )  # Upper bound for H (hue), L (lightness), S (saturation)

    # Create a mask for the green areas
    mask = cv.inRange(cv.cvtColor(image, cv.COLOR_BGR2HSV), lower_green, upper_green)

    # green_layer = image[:, :, 1]
    # _, mask = cv.threshold(green_layer, 80, 1, cv.THRESH_BINARY)
    erosion_kernel = np.ones((13, 13))
    mask = cv.erode(mask, erosion_kernel, iterations=3)
    mask = cv.dilate(mask, erosion_kernel, iterations=7)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv.contourArea)
        cv.drawContours(mask, [c], -1, color=1, thickness=cv.FILLED)

    mask = np.expand_dims(mask, axis=-1)

    # cv.imshow("new mask", np.uint8(mask * image))
    # cv.waitKey(0)
    return np.uint8(mask * image)


def remove_players(image: np.ndarray, im_size: int = 1280) -> list:
    detected_players = player_model.predict(image, classes=[0], imgsz=im_size, conf=0.1)
    boxes = detected_players[0].boxes.xyxyn.tolist()

    return boxes


def find_lines(
    image: np.ndarray,
    size: tuple[int, int] = (480, 270),
    threshold1: int = 30,
    threshold2: int = 50,
    border: int = 10,
    min_votes: int = 1,
    players_im_size: int = 1280,
) -> tuple[np.ndarray, np.ndarray]:
    player_boxes = remove_players(image, im_size=players_im_size)
    result = mask_background(image)

    height, width, _ = image.shape
    scale_height, scale_width = size[1] / height, size[0] / width

    imS = cv.resize(result, size)
    dst = cv.Canny(imS, threshold1=threshold1, threshold2=threshold2)
    dst[-50:, :] = np.zeros(shape=(50, size[0])).astype("uint8")

    for box in player_boxes:
        x0, y0, x1, y1 = (
            int(box[0] * dst.shape[1] - border),
            int(box[1] * dst.shape[0] - border),
            int(box[2] * dst.shape[1] + border),
            int(box[3] * dst.shape[0] + border),
        )
        dst[y0:y1, x0:x1] = 0
        imS[y0:y1, x0:x1] = 0

    try:
        lines = cv.HoughLinesWithAccumulator(
            dst, lines=None, rho=1, theta=np.pi / 180, threshold=min_votes
        ).reshape(-1, 3)
        votes = lines[:, -1]
        cartesian_lines = np.apply_along_axis(
            lambda line: polar2cartesian(line), axis=1, arr=lines
        )
        return clean_lines(
            cartesian_lines=cartesian_lines,
            votes=votes,
            shape=size,
            scale_height=scale_height,
            scale_width=scale_width,
            threshold=20,
        )
    except Exception as e:
        print(e)
        return np.zeros(shape=(0, 2)).astype(np.float32), np.zeros(shape=(0, 1)).astype(
            np.float32
        )


def filter_vertical(lines: np.ndarray, limit: float = 0.2) -> tuple[np.ndarray, float]:
    gradients = lines[:, 0].tolist()
    vertical = [abs(grad) > limit for grad in gradients]
    return lines[np.where(vertical)]


def filter_horizontal(lines: np.ndarray, limit: float = 1) -> tuple[np.ndarray, float]:
    gradients = lines[:, 0].tolist()
    vertical = [abs(grad) <= limit for grad in gradients]
    return lines[np.where(vertical)]


def make_homography(matches, kp, offset):
    src_points = np.float32([kp[m.queryIdx + offset].pt for m in matches]).reshape(
        -1, 1, 2
    )
    dst_points = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, _ = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
    return homography


def warp_point(point: np.ndarray, H: np.ndarray) -> np.ndarray:
    homogenous = (H @ point.reshape((3, 1))).T
    return homogenous / homogenous[:, -1]


def new_project_lines(
    lines: np.ndarray, H: np.ndarray, shape: tuple[int, int] = (1080, 1920)
) -> np.ndarray:
    if lines.shape[0] < 1:
        return lines

    line_points = np.apply_along_axis(
        lambda row: cartesian2points(row, shape=shape), axis=1, arr=lines
    )

    homogenous_points = [
        [line[0].tolist() + [1], line[1].tolist() + [1]]  # NOQA
        for line in line_points
    ]
    line_points = np.float32(homogenous_points)
    warped_points = np.squeeze(
        np.apply_along_axis(lambda x: warp_point(x, H), axis=2, arr=line_points), axis=2
    )[:, :, :-1]
    cartesian_lines = np.array([points2cartesian(row) for row in warped_points])
    return cartesian_lines


def blend_images(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    erosion_kernel = np.ones((5, 5), np.uint8)
    _, im1_mask = cv.threshold(
        cv.cvtColor(image1, cv.COLOR_BGR2GRAY), 1, 1, cv.THRESH_BINARY
    )
    im1_mask = cv.erode(im1_mask, erosion_kernel)

    _, im2_mask = cv.threshold(
        cv.cvtColor(image2, cv.COLOR_BGR2GRAY), 1, 1, cv.THRESH_BINARY
    )

    both_mask = cv.bitwise_and(im1_mask, im2_mask) * 0.5
    im1_only_mask = cv.bitwise_and(im1_mask, cv.bitwise_not(im2_mask))
    im2_only_mask = cv.bitwise_and(im2_mask, cv.bitwise_not(im1_mask))

    im1_only_mask = np.expand_dims(im1_only_mask, axis=-1)
    im2_only_mask = np.expand_dims(im2_only_mask, axis=-1)
    both_mask = np.expand_dims(both_mask, axis=-1)

    result = (
        im1_only_mask * image1
        + both_mask * image1
        + image2 * both_mask
        + im2_only_mask * image2
    )
    return np.uint8(result)


def blend_all_images(image_list: list[np.ndarray]) -> np.ndarray:
    erosion_kernel = np.ones((5, 5), np.uint8)
    image_masks = [
        cv.erode(
            cv.threshold(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 1, 1, cv.THRESH_BINARY)[
                1
            ],
            erosion_kernel,
        )
        for image in image_list
    ]
    weighting_mask = np.sum(image_masks, axis=0)
    weighting_mask[weighting_mask == 0] = 1  # Avoid zero divide

    blended_image = np.zeros_like(image_list[0], dtype=np.float32)

    for i, mask in tqdm(enumerate(image_masks)):
        weighted_mask = mask / weighting_mask
        mask_3_channel = np.stack([weighted_mask, weighted_mask, weighted_mask], axis=2)
        blended_image += mask_3_channel * image_list[i].astype(np.float32)

    return np.clip(blended_image, 0, 255).astype(np.uint8)


def find_homography(
    dest_features: tuple[list[cv.KeyPoint], np.ndarray],
    new_features: tuple[list[cv.KeyPoint], np.ndarray],
    min_matches: int = 150,
    thresh: float = 0.6,
) -> tuple[np.ndarray, int]:
    dest_kp, dest_descr = dest_features
    kp, descr = new_features
    matches = match_descriptors(descr, dest_descr, thresh=thresh)
    if len(matches) < min_matches:
        return None, None

    src_points = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([dest_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, _ = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
    return homography, len(matches)


def GBFIS(
    all_frames: list[np.ndarray],
    min_matches: int = 100,
    start_idx: int = 0,
    threshold: float = 0.6,
    frames: bool | None = None,
) -> np.ndarray:
    frame_bw = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in all_frames]
    center_idx = start_idx
    sift = cv.SIFT_create()
    sift_features = [sift.detectAndCompute(im, None) for im in tqdm(frame_bw)]

    all_centering_homographies = [{"H": np.eye(3), "idx": center_idx}]

    stitch_parents = [{"H": np.eye(3), "idx": center_idx}]
    stitch_options = [i for i in range(len(all_frames)) if i != center_idx]
    pbar = tqdm(total=len(stitch_options))
    while stitch_parents:
        parent = stitch_parents.pop(0)
        parent_features = sift_features[parent["idx"]]

        for i in stitch_options:
            new_features = sift_features[i]

            hom, matches = find_homography(
                parent_features, new_features, min_matches=min_matches, thresh=threshold
            )
            if hom is None:
                continue
            if matches >= min_matches:
                centering_homography = parent["H"] @ hom

            stitch_options.remove(i)
            pbar.update(1)
            stitch_parents.append({"H": centering_homography, "idx": i})
            all_centering_homographies.append({"H": centering_homography, "idx": i})
    pbar.close()
    return all_centering_homographies


def find_im_corners(frame_shape: tuple[int, int], H: np.ndarray) -> np.ndarray:
    h1, w1, _ = frame_shape
    original_size = h1 * w1

    im1_corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corner_locations = cv.perspectiveTransform(im1_corners, H)
    (xmin, ymin) = np.int32(np.floor(corner_locations.min(axis=0).ravel()))
    (xmax, ymax) = np.int32(np.ceil(corner_locations.max(axis=0).ravel()))
    im_size = (xmax - xmin) * (ymax - ymin)
    if abs(im_size) < original_size * 10:
        return corner_locations
    return im1_corners


def find_intersection(
    line1: list[float, float], line2: list[float, float]
) -> tuple[int, int] | None:
    m1, c1 = line1
    m2, c2 = line2
    try:
        x = int((c2 - c1) / (m1 - m2))
    except (RuntimeWarning, ValueError):
        x = 1e9
    y = int(m1 * x + c1)
    return (x, y)  # (y, x) is row, col format


def find_center_score(lines: np.ndarray, frame_size=(1080, 1920)) -> float:
    vertical_lines = filter_vertical(np.copy(lines), 3)
    if len(vertical_lines) == 0:
        return np.inf

    line_points = np.apply_along_axis(
        lambda row: cartesian2points(row, shape=frame_size), axis=1, arr=vertical_lines
    )

    line_starts, line_ends = line_points[:, 0, :], line_points[:, 1, :]
    line_start_x = line_starts[:, 0]
    line_end_x = line_ends[:, 0]

    half_width = frame_size[1] / 2

    line_center_score = (
        np.abs(line_start_x - half_width) + np.abs(line_end_x - half_width)
    ) / 2

    ordering = np.abs(vertical_lines[:, 0]).argsort()[::-1]

    return line_center_score[ordering[0]]


def centeredness(
    cleaned_lines: np.ndarray, frame_size: tuple[int, int] = (1080, 1920)
) -> tuple[list[int], int | None, list[int], float]:
    vertical_lines = filter_vertical(np.copy(cleaned_lines), 0.2)
    if len(vertical_lines) == 0:
        return [], None, [], np.inf
    index_mapping = [
        np.where((line == cleaned_lines).all(axis=1))[0][0] for line in vertical_lines
    ]

    line_points = np.apply_along_axis(
        lambda row: cartesian2points(row, shape=frame_size), axis=1, arr=vertical_lines
    )

    line_starts, line_ends = line_points[:, 0, :], line_points[:, 1, :]
    line_start_x = line_starts[:, 0]
    line_end_x = line_ends[:, 0]

    half_width = frame_size[1] / 2

    line_center_score = (
        np.abs(line_start_x - half_width) + np.abs(line_end_x - half_width)
    ) / 2

    ordering = np.abs(vertical_lines[:, 0]).argsort()[::-1]

    ordered_lines = vertical_lines[ordering]

    slopes = np.copy(ordered_lines[:, 0])
    abs_slopes = np.abs(slopes).reshape(-1, 1)
    dist_mat = pdist(abs_slopes, "euclidean")
    dist_square = squareform(dist_mat)

    # Step 3: Find pairs where the difference is below 0.2
    threshold = 0.2
    left_side = []
    right_side = []
    for i in range(len(slopes)):
        # Sort the distances for the i-th line, excluding itself (i.e., set diagonal to inf)
        distances = np.copy(dist_square[i])
        distances[i] = np.inf
        sorted_idx = np.argsort(distances)

        best_match = sorted_idx[0]

        # Get the distance ratio (best/second-best)
        best_dist = distances[best_match]

        intersection = find_intersection(
            ordered_lines[i, :], ordered_lines[best_match, :]
        )
        center_location = (
            intersection[0],
            slopes[0] * intersection[0] + ordered_lines[0, 1],
        )
        intersect_dist = np.linalg.norm(
            np.array(intersection) - np.array(center_location)
        )
        if (
            best_dist < threshold
            and (slopes[i] * slopes[best_match]) < 0
            and intersect_dist < 1500
            and i not in left_side
            and best_match not in left_side
            and i not in right_side
            and best_match not in right_side
        ):
            if slopes[i] < 0:
                right_side.append(best_match)
                left_side.append(i)
            else:
                right_side.append(i)
                left_side.append(best_match)

    center_score = line_center_score[ordering[0]]

    ordering = ordering.tolist()
    return (
        [index_mapping[ordering[i]] for i in left_side],
        index_mapping[ordering[0]],
        [index_mapping[ordering[i]] for i in right_side],
        center_score,
    )


def center_all(
    all_homographies: dict[int, np.ndarray],
    all_projected_lines: list[np.ndarray],
    center_frame_idx: int,
    frame_size: tuple[int, int],
) -> tuple[dict[int, np.ndarray], list[np.ndarray]]:
    centering_homography = np.linalg.inv(all_homographies[center_frame_idx])
    all_centered_homographies = {
        idx: centering_homography @ H for idx, H in all_homographies.items()
    }

    centered_lines = new_project_lines(
        all_projected_lines, centering_homography, frame_size
    )

    return all_centered_homographies, centered_lines


def get_horizontal(
    cleaned_lines: np.ndarray,
    frame: np.ndarray,
    center_line_gradient: float,
    limit: float = 0.03,
) -> list[int]:
    perpendicular_gradient = -(1 / center_line_gradient)
    masked_im = mask_background(frame)
    try:
        minimum_intercept = min(np.where((masked_im[:, 0] > 0).all(axis=1))[0])
        horizontal_indices = np.where(
            np.all(
                np.array(
                    [
                        np.abs(cleaned_lines[:, 0] - perpendicular_gradient) < limit,
                        cleaned_lines[:, 1] > minimum_intercept,
                    ]
                ),
                axis=0,
            )
        )[0].tolist()

        return horizontal_indices
    except:
        return []


def get_consensus_lines(
    all_homographies: dict[int, np.ndarray],
    all_projected_lines: list[np.ndarray],
    frame_size: tuple[int, int],
    min_votes: float = 0.05,
) -> tuple[list, int | None, list, int]:
    left_votes = []
    center_votes = []
    right_votes = []
    max_rays = 0

    center_score = {}

    for i in all_homographies:
        local_homography = np.linalg.inv(all_homographies[i])
        cleaned_local_lines = new_project_lines(
            all_projected_lines, local_homography, frame_size
        )
        left_side, center_idx, right_side, score = centeredness(cleaned_local_lines)
        max_rays = max(max_rays, len(left_side), len(right_side))
        if center_idx is not None:
            left_votes.append(left_side)
            center_votes.append(center_idx)
            right_votes.append(right_side)
            center_score[i] = (center_idx, score)

    if len(center_votes) == 0:
        return [], None, [], 0

    center = Counter(center_votes).most_common(1)[0][0]

    results = [
        (score, idx)
        for idx, (center_id, score) in center_score.items()
        if center_id == center
    ]

    center_frame_idx = sorted(results)[0][1]

    final_left = []
    final_right = []
    minimum_votes = int(min_votes * len(all_homographies))
    for i in range(max_rays):
        left_line = Counter(
            [item[i] for item in left_votes if i < len(item)]
        ).most_common(1)[0]
        right_line = Counter(
            [item[i] for item in right_votes if i < len(item)]
        ).most_common(1)[0]
        if left_line[1] >= minimum_votes:
            final_left.append(left_line[0])
        if right_line[1] >= minimum_votes:
            final_right.append(right_line[0])
    return final_left, center, final_right, center_frame_idx


def get_formatted_lines(
    centered_lines: np.ndarray,
    horizontal_lines: list[int],
    left_lines: list[int],
    center: int,
    right_lines: list[int],
    frame_size: tuple[int, int],
    close_tol: float = 0.2,
) -> tuple[dict, dict]:
    horizontal_center_y = centered_lines[horizontal_lines, :] @ np.array(
        [(frame_size[1] / 2), 1]
    )

    ordering = horizontal_center_y.argsort()
    ordered_horizontal_center_y = horizontal_center_y[ordering]

    close_dist = frame_size[0] * close_tol

    # If the algorithm managed to detect the top dashed line, then it will be very close to the top line.
    detected_top = sum(
        ordered_horizontal_center_y[1:] < (ordered_horizontal_center_y[0] + close_dist)
    )

    horizontal_line_options = {
        key: None
        for key in (
            "top",
            "five_meter",
            "ten_meter",
            "bottom_ten_meter",
            "bottom_five_meter",
            "bottom",
        )
    }

    match detected_top:
        case 0:
            found_lines = ("top", "bottom_ten_meter", "bottom_five_meter", "bottom")
        case 1:
            found_lines = (
                "top",
                "ten_meter",
                "bottom_ten_meter",
                "bottom_five_meter",
                "bottom",
            )
        case 2:
            found_lines = (
                "top",
                "five_meter",
                "ten_meter",
                "bottom_ten_meter",
                "bottom_five_meter",
                "bottom",
            )
        case _:
            pass  # TODO: DECIDE WHAT TO DO IF THERE ARE FALSE POSITIVES

    for i, option in enumerate(found_lines):
        if i < len(horizontal_lines):
            horizontal_line_options[option] = centered_lines[horizontal_lines, :][
                ordering
            ][i].tolist()

    vertical_line_options = {key: None for key in range(0, 110, 10)}
    vertical_line_options[50] = centered_lines[center].tolist()
    for i, (left_vert_line, right_vert_line) in enumerate(
        zip(left_lines, right_lines, strict=False)
    ):
        vertical_line_options[40 - i * 10] = centered_lines[left_vert_line].tolist()
        vertical_line_options[60 + i * 10] = centered_lines[right_vert_line].tolist()

    return horizontal_line_options, vertical_line_options


def find_intersection_grid(
    horizontal_detected: dict[str, list | None],
    vertical_detected: dict[int, list | None],
) -> list[list[tuple[int, int]]]:
    all_points = []
    for key in (
        "top",
        "five_meter",
        "ten_meter",
        "bottom_ten_meter",
        "bottom_five_meter",
        "bottom",
    ):
        val = horizontal_detected[key]
        if val is None:
            all_points.append([None] * 10)
        else:
            intersections = []
            for dist, point in vertical_detected.items():
                if point is None:
                    intersections.append(None)
                else:
                    intersections.append(find_intersection(val, point))
            all_points.append(intersections)
    return all_points


def find_registering_homography(
    detected_grid: list[list[tuple[int, int]]], method: str = "RANSAC"
) -> np.ndarray:
    true_grid = get_true_intersection_grid()
    matches = []
    for i, row in enumerate(detected_grid):
        for j, point in enumerate(row):
            if point is None:
                continue
            matches.append([point, true_grid[i][j]])
    src_points = np.float32([m for m, _ in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([m for _, m in matches]).reshape(-1, 1, 2)
    match method:
        case "RANSAC":
            homography, _ = cv.findHomography(src_points, dst_points, cv.RANSAC, 100)
        case "LS":
            homography, _ = cv.findHomography(src_points, dst_points, 0)
    return homography


def new_find_registering_homography(
    centered_lines: np.ndarray,
    horizontal_lines: list[int],
    left_lines: list[int],
    center: int,
    right_lines: list[int],
    frame_size: tuple[int, int],
) -> np.ndarray:
    horizontal_center_y = centered_lines[horizontal_lines, :] @ np.array(
        [(frame_size[1] / 2), 1]
    )

    ordering = horizontal_center_y.argsort()
    ordered_horizontal_center_y = horizontal_center_y[ordering]

    line_keys = (
        "top",
        "five_meter",
        "ten_meter",
        "bottom_ten_meter",
        "bottom_five_meter",
        "bottom",
    )

    for i in range(2, min(len(line_keys) + 1, 7)):
        line_combos = combinations(ordering, i)
        for line_combination in line_combos:
            all_line_placements = combinations(range(6), i)
            for placement in all_line_placements:
                horizontal_line_options = {key: None for key in line_keys}
                for line, place in zip(line_combination, placement, strict=False):
                    horizontal_line_options[line_keys[place]] = centered_lines[
                        horizontal_lines, :
                    ][ordering][line].tolist()
                print(horizontal_line_options)


def define_field() -> np.ndarray:
    play_length = 2000
    play_width = 1360
    goal_length = 8 * 20  # CAN BE 6m TO 11m BOTH SIDES - BUT NRL PREFERS 8M
    main_line_width = 3
    small_line_width = 2

    field = np.zeros((play_width, play_length + 2 * goal_length))

    field[:3, :] = 1
    field[-3:, :] = 1
    field[:, :3] = 1
    field[:, -3:] = 1
    for j, i in enumerate(range(goal_length, play_length + goal_length + 1, 200)):
        field[:, (i - 1) : (i + 1)] = 1
        if 0 < j < 10:
            field[[199, 399, play_width - 199, play_width - 399], :] = 1
    return field


def get_true_intersection_grid() -> list[list[tuple[int, int]]]:
    play_length = 2000
    play_width = 1360
    goal_length = 8 * 20  # CAN BE 6m TO 11m BOTH SIDES - BUT NRL PREFERS 8M
    horizontal_lines = [0, 199, 399, play_width - 199, play_width - 399, play_width - 1]
    vertical_lines = list(range(goal_length, play_length + goal_length + 1, 200))
    return [[(ver, hor) for ver in vertical_lines] for hor in horizontal_lines]


def get_true_field_lines() -> np.ndarray:
    play_length = 2000
    play_width = 1360
    goal_length = 8 * 20  # CAN BE 6m TO 11m BOTH SIDES - BUT NRL PREFERS 8M
    horizontal_lines = np.array(
        [
            [[h, 0], [h, play_length + 2 * goal_length]]
            for h in [0, 200, 400, play_width - 200, play_width - 400, play_width]
        ]
    )
    vertical_lines = np.array(
        [
            [[0, v], [play_width, v]]
            for v in range(goal_length, play_length + goal_length + 1, 200)
        ]
    )
    return np.concatenate([horizontal_lines, vertical_lines])


def register_all_frames(
    frames: list[np.ndarray],
    registration: np.ndarray,
    all_centered_homographies: dict[int, np.ndarray],
    frame_size: tuple[int, int] = (1920, 1080),
    field_size: tuple[int, int] = (2320, 1360),
    overlay: bool = True,
    rgb_template: np.ndarray | None = None,
    inverse: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    all_registered_homographies = {
        idx: registration @ H for idx, H in all_centered_homographies.items()
    }
    registered_frames = {}

    for i in all_registered_homographies:
        if not inverse:
            registered_frame = cv.warpPerspective(
                frames[i], all_registered_homographies[i], field_size
            )
            if overlay:
                registered_frame = cv.add(rgb_template, registered_frame)
        else:
            registered_lines = cv.warpPerspective(
                rgb_template, np.linalg.inv(all_registered_homographies[i]), frame_size
            )
            if overlay:
                registered_frame = cv.add(registered_lines, np.copy(frames[i]))
        registered_frames[i] = registered_frame

    return registered_frames, all_registered_homographies


def show_results(
    img: np.ndarray,
    lines1: np.ndarray,
    lty: str = "cartesian",
    filetype: str = "python",
) -> None:
    for j in range(len(lines1)):
        match lty:
            case "polar":
                rho = lines1[j][0]
                theta = lines1[j][1]

                a = np.cos(theta)
                b = np.sin(theta)

                x0 = a * rho
                y0 = b * rho

                pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
                pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
            case "points":
                x0, y0, x1, y1 = lines1[j].flatten()
                pt1 = (int(x0), int(y0))
                pt2 = (int(x1), int(y1))
            case "cartesian":
                m = lines1[j][0]
                c = lines1[j][1]

                x0 = 0
                y0 = c
                x1 = 10000
                y1 = m * x1 + c

                pt1 = (int(x0), int(y0))
                pt2 = (int(x1), int(y1))
        try:
            cv.line(img, pt1, pt2, (0, 255, 0), 2, cv.LINE_AA)
        except:
            pass
    match filetype:
        case "notebook":
            # display(Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)))
            pass
        case "python":
            cv.imshow("Results", img)
            cv.waitKey(0)
        case "manual":
            cv.imshow("Results", img)
