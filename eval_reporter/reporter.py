""" reporter.py

    This is intended a as a simple drop-in evaluation analyzer for tensorflow.

    You can create a `tf_reporter` layer which accepts the following tensors:
    1. image:     [batch x W x H x 3]
    2. predicted: [batch]
    3. expected:  [batch]

    The `tf_reporter` is abstracted within the `EvalReporter` class which can
    be queried for the tf `op` which can be inserted into the evaluation
    session's execution graph.

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

################################################################################

class EvalReporter(object):
    """ EvalReporter is a class which is constructed with a given batches set of
        tensors for the image, predictions and the expectation.
    """

    failure_histogram = None        # incorrectly predicted classes
    success_histogram = None        # correctly predicted classes
    op                = None        # `py_func` layer which dumps images


    def __init__(self, images=[], predicted=[], expected=[]):
        """ constructor for the evaluation reporter.
            TODO: The lengths of the incoming tensors must match!
        """
        self.clear()
        self.op = tf.py_func(self._pyfunc, [images, actual, expected], tf.float32)


    def _pyfunc(images, actual, expected):
        """ _pyfunc is the python_func's "op" which will accept a list of the
            images, predicted classes and the expectations.  These at this point
            are NOT tensors, but are `numpy.ndarray`'s.
        """
        total = 0
        error = 0
        for i in range(len(images)):
            im = scipy.misc.toimage(images[i])              # ndarray -> PIL.Image
            bs = cStringIO.StringIO()                       # buffer to hold image
            im.save(bs, format="JPEG")                      # PIL.Image -> JPEG
            b64s = base64.b64encode(bs.getvalue())          # JPEG -> Base64
            total += 1
            if expected[i] == actual[i]:
                if actual[i] in self.success_histogram:
                    self.success_histogram[actual[i]].append(b64s)
                else:
                    self.success_histogram[actual[i]] = [b64s]
            else:
                error += 1
                if actual[i] in self.failure_histogram:
                    self.failure_histogram[actual[i]].append([b64s, expected[i]])
                else:
                    self.failure_histogram[actual[i]] = [[b64s, expected[i]]]

        return np.float32(((total - error)/total) if total > 0 else 0)


    def get_op(self):
        """ get_op returns the tensorflow wrapped `py_func` which will convert the local
            tensors into numpy ndarrays.
        """
        return self.op


    def clear(self):
        """ clear resets the histograms for `this` reporter.
        """
        self.failure_histogram = dict()
        self.success_histogram = dict()


    def write_html_file(self, file_path):
        """ write_html_file dumps the current histograms to the specified `file_path`.
        """
        html = head
        html += "  <h1>Correct Predictions</h1><br>\n"
        for k in self.success_histogram:
            html += "  <h3>Class {}</h3><br>\n".format(k)
            files = self.success_histogram[k]
            for f in files:
                html += """  <img src="data:image/jpeg;base64,{}" title="class:{}"/>""".format(f, k)
        html += "  <hr><br><br>\n"
        html += "  <h1>Incorrect Predictions</h1><br>\n"
        for k in self.failure_histogram:
            html += "  <h3>Class {}</h3><br>\n".format(k)
            items = self.failure_histogram[k]
            for it in items:
                f, actual = it[0], it[1]
                html += """  <img src="data:image/jpeg;base64,{}" title="pred:{} exp:{}"/>""".format(f, k, actual)
        html += tail
        with open(file_path, "w") as fout:
            fout.write(html)

################################################################################
