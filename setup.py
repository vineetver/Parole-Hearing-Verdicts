from setuptools import setup, find_packages

setup(name='Texas_Parole_Verdicts',
      version='1.0',
      description='This package contains the code for the Texas Parole Verdicts Repository',
      author='VineetVerma',
      author_email='vineetver@hotmail.com',
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'pandas',
          'numpy',
          'matplotlib',
          'sklearn',
          'pandas',
          'scikit-learn',
          's3fs',
          'pyarrow',
          'boto3',
          'sodapy',
          'tensorflow'
      ],
      entry_points={
          'console_scripts': [
              'scrape_data = main.scrape:main',
              'clean_data = main.clean:main',
              'feature_engineering = main.feature_engineering:main',
              'train_model = main.train_model:main',
          ],
      }
      )
