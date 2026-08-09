from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'captioner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='docker',
    maintainer_email='nwzantout@gmail.com',
    description='The captioning package.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'captioning_node = captioner.captioning_node:main',
            'caption_crops = captioner.models.captioning:main',
            'qwen_vqa = captioner.models.vqa:main',
            'qwen_vqa_server = captioner.qwen_vqa_server:main',
            'qwen_vqa_ask = captioner.qwen_vqa_client:main',
            'qwen_vqa_wait_ready = captioner.qwen_vqa_wait_ready:main',
            'fetch_weights = captioner.tools.fetch_weights:main',
        ],
    },
)
