# setup.py - Project Setup Configuration

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tech-career-predictor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered tech career prediction system using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/tech-career-predictor",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/tech-career-predictor/issues",
        "Documentation": "https://github.com/your-username/tech-career-predictor/wiki",
        "Source Code": "https://github.com/your-username/tech-career-predictor",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: Flask",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-flask>=1.2.0",
            "pytest-cov>=4.1.0",
            "coverage>=7.2.7",
            "flake8>=6.0.0",
            "black>=23.7.0",
            "isort>=5.12.0",
        ],
        "deploy": [
            "gunicorn>=21.2.0",
            "boto3>=1.28.25",
            "psutil>=5.9.5",
        ],
        "monitoring": [
            "redis>=4.6.0",
            "celery>=5.3.1",
            "SQLAlchemy>=2.0.19",
        ],
    },
    entry_points={
        "console_scripts": [
            "career-predictor=app:main",
            "train-model=src.model_training:main",
            "preprocess-data=src.data_preprocessing:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["templates/*.html", "static/css/*.css", "static/js/*.js", "static/images/*"],
        "models": ["*.pkl"],
        "data": ["*.csv"],
    },
    zip_safe=False,
    keywords=[
        "machine learning",
        "career prediction",
        "flask",
        "artificial intelligence",
        "personality assessment",
        "tech careers",
        "random forest",
        "web application",
        "data science",
        "career guidance",
    ],
    platforms=["any"],
)