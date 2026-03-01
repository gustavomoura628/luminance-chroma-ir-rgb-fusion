from setuptools import find_packages, setup

package_name = "luminance_chroma_ir_rgb_fusion"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/params.yaml"]),
        ("share/" + package_name + "/launch", ["launch/fusion.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gus",
    maintainer_email="gus@todo.todo",
    description="IR luma + RGB chroma fusion for Intel RealSense D435i trinocular setup",
    license="MIT",
    entry_points={
        "console_scripts": [
            "fusion_node = luminance_chroma_ir_rgb_fusion.fusion_node:main",
        ],
    },
)
