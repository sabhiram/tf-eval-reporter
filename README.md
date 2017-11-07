# tf-eval-reporter

Tensorflow "layer" that categorizes good and bad predictions into a HTML document

## Install

```shell
git clone https://github.com/sabhiram/tf-eval-reporter.git
cd tf-eval-reporter
python setup.py install
```

## Usage

This assumes that you have access to a data provider to get the inputs to process
and the expected outputs, as well as a model to evaluate the inputs against to get
a batch-wide set of results (The `ModelFunction` below is such an abstraction).

```py
from eval_reporter import EvalReporter

[image, label] = provider.get(['image', 'label'])

input, expected = tf.train.batch(
	[image, label],
	batch_size=32)

output = ModelFunction(input)

# Create a reporter based on the input, output and expectation tensors.
reporter = EvalReporter(input, output, expected)
# Or more commonly
# reporter = EvalReporter(images, predictions, labels)

# Register the reporter op if you want to enable the report building.
# This is usually done by passing this op in addition to the rest of the 
# ops that you want to process during evaluation.
eval_ops = [reporter.get_op()]

slim.evaluation.evaluate_once(
	master=...,
	checkpoint_path=...,
	logdir=...,
	num_evals=...,
	eval_op=eval_ops,
	vairables_to_restore=...)

# Finally, to get out a HTML document of this run...
reporter.write_html_file("eval.html")
```

The above usage is mostly broken and serves as a gentle guide to integrating this into your own deep learning evaluation flow.  The idea here is not to have this tied down to a single framework, but rather allow it to accept input of the format [`image tensor`, `prediction tensor`, `expectation tensor`]. 

## TODO

1. Verify all input tensors are of same width.
2. Make the HTML document prettier and easier to filter based on classes.
3. Add ability for reporter to map the items in the expected and predicted labels to string labels.
