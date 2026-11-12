from glob import glob

from setuptools import setup

package_name = "hsl_behavior"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bami001",
    maintainer_email="gntjr2@gmail.com",
    description="HSL behavior (strategy + skills)",
    license="BSD-3-Clause",
    entry_points={"console_scripts": ["behavior = hsl_behavior.behavior_node:main"]},
)
