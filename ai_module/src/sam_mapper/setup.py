from setuptools import setup
import os
from glob import glob

package_name = 'sam_mapper'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, f'{package_name}.tools'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CMU-VLN Team',
    maintainer_email='team@example.com',
    description='SAM 3 / SAM 3.1 driven 3D open-vocabulary instance mapping.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sam_node = sam_mapper.sam_node:main',       # SAM3 2D detector node
            'map_node = sam_mapper.map_node:main',       # lidar fusion / 3D mapping node
            'sam3_probe = sam_mapper.sam3_backend:main',
        ],
    },
)
