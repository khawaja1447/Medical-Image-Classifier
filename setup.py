from setuptools import find_packages, setup

setup(
    name="medical-image-classifier",
    version="1.0.0",
    author="Abeer Ashraf",
    description="ResNet-50 chest X-ray classifier with Grad-CAM explainability",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "opencv-python-headless>=4.8.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.28.0",
        "PyYAML>=6.0",
    ],
)
