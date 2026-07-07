from setuptools import find_packages, setup

package_name = 'driverless'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='URC',
    maintainer_email='-',
    description='Driverless car package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drive_forward = driverless.drive_forward:main',
            'centerline_node = driverless.path_planning.centerline_node:main',
            'rrt_node = driverless.path_planning.rrt_node:main',
            'pure_pursuit = driverless.controller.pure_pursuit:main',
        ],
    },
)
