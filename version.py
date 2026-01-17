import tensorflow as tf

from pathlib import Path
import sys


if __name__ == "__main__":
    # TensorFLow
    print(tf.__version__)
    # print(tf.reduce_sum(tf.random.normal([1000, 1000])))

    ROOT = Path().resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    print(sys.path)
