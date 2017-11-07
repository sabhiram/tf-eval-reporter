from setuptools import setup

setup(name="eval_reporter",
      version="0.1",
      description="tensorflow correctness reporter",
      url="https://github.com/sabhiram/tf-eval-reporterer",
      author="sabhiram",
      install_requires=[
        "scipy",
        "numpy",
        "pillow",
        "jinja2",
        # "tensorflow",
      ],
      packages=["eval_reporter"])
