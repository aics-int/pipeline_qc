from setuptools import find_packages, setup

PACKAGE_NAME = 'pipeline_qc'


"""
Notes:
MODULE_VERSION is read from pipeline_qc/version.py.
See (3) in following link to read about versions from a single source
https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version
"""

MODULE_VERSION = ""
exec(open(PACKAGE_NAME + "/version.py").read())


def readme():
    with open('README.md') as f:
        return f.read()


test_deps = ['pytest', 'pytest-cov', 'pytest-raises']

lint_deps = ['flake8']
interactive_dev_deps = [
    # -- Add libraries/modules you want to use for interactive
    # -- testing below (e.g. jupyter notebook).
    # -- E.g.
    # 'matplotlib>=2.2.3',
    # 'jupyter',
    # 'itkwidgets==0.12.2',
    # 'ipython==7.0.1',
    # 'ipywidgets==7.4.1'
    'bump2version'
]
all_deps = [*test_deps, *lint_deps, *interactive_dev_deps]

extras = {
    'test': test_deps,
    'lint': lint_deps,
    'interactive_dev': interactive_dev_deps,
    # These are for legacy compatibility with the gradle build setup
    'test_group': test_deps,
    'lint_group': lint_deps,
    'interactive_dev_group': interactive_dev_deps,
    'all': all_deps
}

setup(name=PACKAGE_NAME,
      version=MODULE_VERSION,
      description='pipeline qc methods',
      long_description=readme(),
      author='AICS',
      author_email='calystay@alleninstitute.org',
      license='Allen Institute Software License',
      packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
      entry_points={
          "console_scripts": [
              "fov_qc_cli={}.bin.fov_qc_cli:main".format(PACKAGE_NAME),
              "cardio_mip_qc_cli={}.bin.cardio_mip_qc_cli:main".format(PACKAGE_NAME),
              "cell_seg_cli={}.bin.cell_seg_cli:main".format(PACKAGE_NAME),
              "labkey_cell_generation={}.bin.labkey_cell_generation:main".format(PACKAGE_NAME),
              "upload_aligned_files={}.bin.aligned_file_upload:main".format(PACKAGE_NAME)
          ]
      },
      install_requires=[
          # List of modules required to use/run this module.
          # -- E.g.
          # 'numpy>=1.15.1',
          # 'requests'
          'aicsfiles',
          'aicsimageio>=3.2.1',
          'aics_dask_utils',
          'bokeh',
          'dask',
          'dask_jobqueue',
          'labkey',
          'lkaccess',
          'matplotlib',
          'numpy',
          'pandas',
          'pyyaml',
          'scipy',
          'scikit-image',
          'tqdm'
      ],

      # For test setup. This will allow JUnit XML output for Jenkins
      setup_requires=['pytest-runner'],
      tests_require=test_deps,

      extras_require=extras,
      zip_safe=False
      )
