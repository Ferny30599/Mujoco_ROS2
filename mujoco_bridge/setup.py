from setuptools import setup
package_name = 'mujoco_bridge'

setup(
    name=package_name,                   # <-- guion BAJO
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='ROS2 <-> MuJoCo control bridge',
    license='Apache-2.0',
    entry_points={
    'console_scripts': [
        'bridge = mujoco_bridge.bridge:main',
        'pub_demo = mujoco_bridge.pub_demo:main',   
        'plot_logger = mujoco_bridge.plot_logger:main',
        ],
    },
)
