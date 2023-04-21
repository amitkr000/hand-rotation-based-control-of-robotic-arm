from setuptools import setup
import os
from glob import glob

package_name = 'cv_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amit',
    maintainer_email='amitkr4538@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	"cam_pub = cv_control.cam_pub:main",
        	"cam_sub = cv_control.cam_sub:main",
        	"handpose_pub = cv_control.handpose_pub:main",
        	"joint = cv_control.joint:main",
        ],
    },
)
