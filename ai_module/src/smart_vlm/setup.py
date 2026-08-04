from setuptools import setup
import os
from glob import glob

package_name = 'smart_vlm'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CMU-VLN Team',
    maintainer_email='team@example.com',
    description='Team AI module: supervisor + exploration + reasoning.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'supervisor = smart_vlm.supervisor:main',
            'bag_fetch = smart_vlm.bag_fetch:main',
            'qwen_numerical = smart_vlm.qwen_numerical:main',
            'category1_reasoner = smart_vlm.category1_reasoner:main',
        ],
    },
)
