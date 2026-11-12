from glob import glob
from setuptools import setup

package_name = "humanoid_sim"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/models", glob("models/*")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="MuJoCo hardware bridge for the mx3 match stack",
    license="Proprietary",
    entry_points={"console_scripts": [
        "sim_gc = humanoid_sim.sim_gc:main",
        "sim_monitor = humanoid_sim.sim_monitor:main",
    ]},
)
