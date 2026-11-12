from setuptools import setup

package_name = "hsl_team_comm"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bami001",
    maintainer_email="gntjr2@gmail.com",
    description="HSL team communication",
    license="BSD-3-Clause",
    entry_points={"console_scripts": ["team_comm = hsl_team_comm.team_comm_node:main"]},
)
