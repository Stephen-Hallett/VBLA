import contextlib
import json
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from natsort import natsorted
from VBLA_utils import *


def main() -> None:
    image_directory = sys.argv[1]

    directory_path = Path(image_directory)
    print(directory_path)

    with contextlib.suppress(FileExistsError):
        (directory_path / "registration").mkdir()

    image_files = natsorted(directory_path.glob("*.jpg"))
    frames_orig = [cv.imread(file) for file in image_files]

    frame_size = frames_orig[0].shape

    all_lines_file = "all_lines.json"

    if all_lines_file not in [
        os.path.basename(file) for file in directory_path.glob("*.json")
    ]:
        all_lines = [find_lines(frame, min_votes=80) for frame in tqdm(frames_orig)]

        with (directory_path / all_lines_file).open("w") as f:
            serialized = [
                (lines.tolist(), votes.tolist()) for lines, votes in all_lines
            ]
            json.dump(serialized, f)
    else:
        with (directory_path / all_lines_file).open("r") as f:
            all_lines_data = json.load(f)
            all_lines = [
                (np.array(lines), np.array(votes)) for lines, votes in all_lines_data
            ]

    all_lines, all_votes = (
        [lines for lines, _ in all_lines],
        [votes for _, votes in all_lines],
    )
    print(all_lines[0].shape)

    if len(sys.argv) > 2:
        start_idx = int(sys.argv[2])
    else:
        initial_center_scores = [
            (find_center_score(lines, frame_size), i)
            if lines.shape[0] > 0
            else (np.inf, i)
            for i, lines in enumerate(all_lines)
        ]
        start_idx = min(initial_center_scores, key=lambda x: x[0])[1]

    print("Detected start index:", start_idx)

    frame_size = frames_orig[0].shape
    homography_file = "homographies.json"

    if homography_file not in [
        os.path.basename(file) for file in directory_path.glob("*.json")
    ]:
        all_homographies = GBFIS(
            frames_orig, min_matches=80, start_idx=start_idx, threshold=0.65
        )
        all_homographies = {item["idx"]: item["H"] for item in all_homographies}

        with (directory_path / homography_file).open("w") as f:
            serialized = {idx: H.tolist() for idx, H in all_homographies.items()}
            json.dump(serialized, f)
    else:
        with (directory_path / homography_file).open("r") as f:
            homography_data = json.load(f)
            all_homographies = {
                int(idx): np.array(H) for idx, H in homography_data.items()
            }

    line_file = "lines.json"

    if line_file not in [
        os.path.basename(file) for file in directory_path.glob("*.json")
    ]:
        all_projected_lines, votes = clean_lines(
            cartesian_lines=np.concatenate(
                [
                    new_project_lines(all_lines[idx], H, frame_size)
                    for idx, H in all_homographies.items()
                    if all_lines[idx].size > 1
                ]
            ),
            votes=np.concatenate([all_votes[idx] for idx in all_homographies]),
            shape=frame_size,
            threshold=100,
        )

        all_projected_lines = all_projected_lines[:30]
        votes = votes[:30]

        with (directory_path / line_file).open("w") as f:
            serialized = all_projected_lines.tolist()
            json.dump(serialized, f)
    else:
        with (directory_path / line_file).open("r") as f:
            all_projected_lines = json.load(f)
            all_projected_lines = np.array(all_projected_lines)

    final_left, center, final_right, center_frame_idx = get_consensus_lines(
        all_homographies, all_projected_lines, frame_size
    )

    all_centered_homographies, centered_lines = center_all(
        all_homographies, all_projected_lines, center_frame_idx, frame_size
    )

    horizontal_lines = get_horizontal(
        centered_lines,
        frames_orig[center_frame_idx],
        centered_lines[center, 0],
        limit=0.05,
    )

    horizontal_detected, vertical_detected = get_formatted_lines(
        centered_lines,
        horizontal_lines,
        final_left,
        center,
        final_right,
        frame_size,
        close_tol=0.2,
    )

    field_template = define_field()
    rgb_template = cv.cvtColor(np.uint8(field_template * 255), cv.COLOR_GRAY2BGR)

    detected_grid = find_intersection_grid(horizontal_detected, vertical_detected)
    print(detected_grid)

    registration = find_registering_homography(detected_grid, method="LS")
    print(registration)

    all_registered_frames, all_registered_homographies = register_all_frames(
        frames_orig,
        registration,
        all_centered_homographies,
        overlay=True,
        rgb_template=rgb_template,
        inverse=False,
    )

    with (directory_path / "registration" / homography_file).open("w") as f:
        serialized = {idx: H.tolist() for idx, H in all_registered_homographies.items()}
        json.dump(serialized, f)

    for i, frame in all_registered_frames.items():
        cv.imwrite(directory_path / "registration" / f"registered_frame{i}.png", frame)

    all_registered_frames_inv, all_registered_homographies_inv = register_all_frames(
        frames_orig,
        registration,
        all_centered_homographies,
        overlay=True,
        rgb_template=rgb_template,
        inverse=True,
    )

    for i, frame in all_registered_frames_inv.items():
        cv.imwrite(
            directory_path / "registration" / f"registered_frame_inv{i}.png", frame
        )


if __name__ == "__main__":
    main()
