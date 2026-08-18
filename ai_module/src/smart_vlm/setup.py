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
    description='Team AI module: mission supervisor, numerical reasoner, eval harness.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'smart_vlm = smart_vlm.smart_vlm:main',
            'bag_fetch = smart_vlm.bag_fetch:main',
            'scene_fetch = smart_vlm.scene_fetch:main',
            'qwen_numerical = smart_vlm.qwen_numerical:main',
            'numerical_reasoner = smart_vlm.numerical_reasoner:main',
            'object_reference_reasoner = smart_vlm.object_reference_reasoner:main',
            'eval_orchestrator = smart_vlm.eval_orchestrator:main',
            'cat1_bench = smart_vlm.cat1_bench:main',
            'cat2_bench = smart_vlm.cat2_bench:main',
            'extract_bench = smart_vlm.extract_bench:main',
            'wait_ready = smart_vlm.wait_ready:main',
        ],
    },
)
