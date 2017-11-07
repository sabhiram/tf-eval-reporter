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
from jinja2 import Template

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
        self.op = tf.py_func(self._pyfunc, [images, predicted, expected], tf.float32)


    def _pyfunc(self, images, predicted, expected):
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
            if expected[i] == predicted[i]:
                if predicted[i] in self.success_histogram:
                    self.success_histogram[predicted[i]].append(b64s)
                else:
                    self.success_histogram[predicted[i]] = [b64s]
            else:
                error += 1
                if predicted[i] in self.failure_histogram:
                    self.failure_histogram[predicted[i]].append([b64s, expected[i]])
                else:
                    self.failure_histogram[predicted[i]] = [[b64s, expected[i]]]

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
        report = Template("""<!DOCTYPE html>
<html>
<head>
    <title>Prediction Analyzer</title>
</head>
<body>
    <h1>Correct Predictions</h1><br>
    {%  for class, images in me.success_histogram.items() %}
        <h3>Class {{ class }}</h3><br>
        {%  for img in images %}
            <img src="data:image/jpeg;base64,{{img}}" title="class:{{class}}" />
        {%  endfor %}
    {%  endfor %}
    <hr><br><br>

    <h1>Incorrect Predictions</h1><br>
    {%  for class, groups in me.failure_histogram.items() %}
        <h3>Class {{ class }}</h3><br>
        {%  for group in groups %}
            <img src="data:image/jpeg;base64,{{group[0]}}" title="pred:{{group[1]}} exp:{{class}}" />
        {%  endfor %}
    {%  endfor %}
    <hr><br><br>
</body>
</html>
""").render(me=self)
        with open(file_path, "w") as fout:
            fout.write(report)

################################################################################
