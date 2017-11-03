""" report.py

    This is intended a as a simple drop-in evaluation analyzer for tensorflow.

    You can create a `tf_report` layer which accepts the following tensors:
    1. image:     [batch x W x H x 3]
    2. predicted: [batch]
    3. expected:  [batch]

    This will return an op that will need to be evaluated alongside the rest
    of the inference graph.  

    TODO: Detailed usage etc.

"""
import scipy
import base64
import cStringIO
import numpy as np
import tensorflow as tf
from PIL import Image

################################################################################

head = """<!DOCTYPE html>
<html>
<head>
  <title>Prediction Analyzer</title>
</head>
<body>"""
tail = """</body>
</html>
"""

hist_err = dict()
hist_ok  = dict()
def histogram_to_file(file_path):
    html = head
    html += "  <h1>Correct Predictions</h1><br>\n"
    for k in hist_ok:
        html += "  <h3>Class {}</h3><br>\n".format(k)
        files = hist_ok[k]
        for f in files:
            html += """  <img src="data:image/jpeg;base64,{}" title="class:{}"/>""".format(f, k)
    html += "  <hr><br><br>\n"
    html += "  <h1>Incorrect Predictions</h1><br>\n"
    for k in hist_err:
        html += "  <h3>Class {}</h3><br>\n".format(k)
        items = hist_err[k]
        for it in items:
            f, actual = it[0], it[1]
            html += """  <img src="data:image/jpeg;base64,{}" title="pred:{} exp:{}"/>""".format(f, k, actual)
    html += tail
    with open(file_path, "w") as fout:
        fout.write(html)


def add_success(cls, img_str):
    if cls in hist_ok:
        hist_ok[cls].append(img_str)
    else:
        hist_ok[cls] = [img_str]


def add_failure(cls, exp, img_str):
    if cls in hist_err:
        hist_err[cls].append([img_str, exp])
    else:
        hist_err[cls] = [[img_str, exp]]

################################################################################

def extract_prediction_func(images, actual, expected):
    total = 0
    error = 0
    for i in range(len(images)):
        im = scipy.misc.toimage(images[i])              # ndarray -> PIL.Image
        bs = cStringIO.StringIO()                       # buffer to hold image
        im.save(bs, format="JPEG")                      # PIL.Image -> JPEG
        b64s = base64.b64encode(bs.getvalue())          # JPEG -> Base64
        total += 1
        if expected[i] == actual[i]:
            add_success(actual[i], b64s)
        else:
            error += 1
            add_failure(actual[i], expected[i], b64s)

    return np.float32(((total - error)/total) if total > 0 else 0)

################################################################################

def get_report(images, actual, expected):
    op = tf.py_func(extract_prediction_func, [images, actual, expected], tf.float32)
    return op

################################################################################