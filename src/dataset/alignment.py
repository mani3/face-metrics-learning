import os
import sys
import logging
import tarfile
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

from mtcnn.mtcnn import MTCNN

IMAGE_SIZE = 128

mtcnn = MTCNN()
logging.basicConfig(filename="batch_alignment.log")


def alignment(filepath: str, size: int):
  image = Image.open(filepath).convert("RGB")

  image_np = np.array(image)
  faces = mtcnn.detect_faces(image_np)

  try:
    box = faces[0]["box"]
    box = (box[0], box[1], box[0] + box[2], box[1] + box[3])
    crop_image = image.crop(box)
    crop_image.thumbnail((size, size), Image.LANCZOS)
  except Exception:
    crop_image = None
  return crop_image


def main(args):
  output_dir = args.output_dir
  with tarfile.open(args.tar_path) as tar:
    for member in tqdm(tar.getmembers()):
      f = tar.extractfile(member)

      if f is None:
        continue

      crop_image = alignment(f, IMAGE_SIZE)
      if crop_image is None:
        logging.error(f"filename={member.name}")
      else:
        dirname, _ = os.path.split(member.name)
        os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)
        output_path = os.path.join(output_dir, member.name)
        crop_image.save(output_path)


if __name__ == "__main__":
  """
  $ python -m src.dataset.alignment \
    --tar_path '/path/to/vggface2_train.tar.gz'
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--tar_path", type=str, help="e.g. /path/to/vggface2_train.tar.gz")
  parser.add_argument("--output_dir", type=str, default="/tmp/vggface2")
  main(parser.parse_args(sys.argv[1:]))
