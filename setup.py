from setuptools import setup, find_packages

setup(
    name='classifytweet',
    version='0.0.1rc0',

    # Package data
    packages=find_packages(),
    include_package_data=True,

    # Insert dependencies list here
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'nltk',
        'Keras',
        'tensorflow',
        'matplotlib',
        'flask',
        'gevent',
        'gunicorn',
        'torch',
        'torchvision',
        'boto3'
    ],

    entry_points={
       "classifytweet.training": [
           "train=classifytweet.train:entry_point",
       ],
        "classifytweet.hosting": [
           "serve=classifytweet.server:start_server",
       ]
    }
)


'''
Traceback (most recent call last):
File "/opt/run.py", line 1, in <module>
from classifytweet.train import entry_point
File "/usr/local/lib/python3.6/site-packages/classifytweet/train.py", line 14, in <module>
from resolve import paths

ModuleNotFoundError: No module named 'resolve'
'''
