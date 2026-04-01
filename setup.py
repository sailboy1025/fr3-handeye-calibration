from setuptools import find_packages, setup

package_name = 'onr_handeye'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/collect_handeye.launch.py']),
        ('share/' + package_name, ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sai',
    maintainer_email='sxj749@case.edu',
    description='Minimal ROS2 hand-eye data collection and base-camera solve pipeline.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'collect_samples = onr_handeye.collect_samples_node:main',
            'auto_collect_samples = onr_handeye.auto_collect_node:main',
            'solve_base_camera = onr_handeye.solve_base_camera:main',
            'check_projection = onr_handeye.check_projection:main',
        ],
    },
)
