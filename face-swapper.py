#!/usr/bin/env python

import argparse

import cv2
import insightface
from insightface.app import FaceAnalysis
from tqdm import tqdm

assert insightface.__version__ >= "0.7"


def get_faces(face_analyzer, image):
    return sorted(face_analyzer.get(image), key=lambda x: x.bbox[0])


def image(path):
    if path is None:
        return None
    return cv2.imread(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("SOURCE_IMAGE", type=image)
    parser.add_argument("TARGET_IMAGE", nargs="?", type=image)
    args = parser.parse_args()
    source_image, target_image = args.SOURCE_IMAGE, args.TARGET_IMAGE

    one_image = target_image is None

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )

    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx", download=True, download_zip=True
    )

    source_faces = get_faces(app, source_image)

    if one_image:
        target_image = source_image.copy()
        target_faces = source_faces.copy()
    else:
        target_faces = get_faces(app, target_image)

    swap_pairs = [
        (i, j)
        for i in range(len(source_faces))
        for j in range(len(target_faces))
        if not one_image or i != j
    ]
    with tqdm(total=len(swap_pairs)) as pbar:
        for i, source_face in enumerate(source_faces):
            res = target_image.copy()
            for j, face in enumerate(target_faces):
                if one_image and i == j:
                    # avoid swapping source face if it is same as target
                    continue
                res = swapper.get(res, face, source_face, paste_back=True)
                pbar.update()
            cv2.imwrite(f"{i:03}_swapped.jpg", res)
