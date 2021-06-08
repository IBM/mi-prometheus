"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
# from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='miprometheus',  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.3.1',  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='Mi-Prometheus: Standardizing interfaces between problems, '
                'models architectures, and training/testing configurations.',  # Required

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',  # Optional (see note above)

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://github.com/IBM/mi-prometheus/',  # Optional
    license='Apache 2.0',

    # This should be your name or the name of the organization which owns the
    # project.
    author='Tomasz Kornuta, Vincent Marois, Ryan L. McAvoy, Younes Bouhadjar, '
           'Alexis Asseman, Vincent Albouy, T.S. Jayram, Ahmet S. Ozcan',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='tkornuta@us.ibm.com',  # Optional

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    # This information is only used for searching & browsing projects on PyPI, not for installing projects
    # Checkout numpy: https://pypi.org/project/numpy/

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        # 'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        # 'License :: OSI Approved :: MIT License',
        # 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Operating System :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='mi-prometheus pytorch machine-learning model problem worker grid-worker',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(exclude=['docs', 'configs', 'build', 'experiments', 'scripts']),  # Required

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    python_requires='~=3.6',
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # Should not pin down version
    # It is not considered best practice to use install_requires to pin
    # dependencies to specific versions, or to specify sub-dependencies
    # (i.e. dependencies of your dependencies). This is overly-restrictive,
    # and prevents the user from gaining the benefit of dependency upgrades.
    install_requires=['torchvision==0.2.2',  # v0.2.0 is needed for the doc build, but we are specifying it in docs/requirements.txt
                      'torchtext',
                      'tensorboardX',
                      'matplotlib',
                      'pandas',
                      'Pillow==8.2.0',
                      'numpy',
                      'psutil',
                      'tqdm==4.19.9',
                      'nltk',
                      'h5py',
                      'sphinx_rtd_theme',
                      'pyqt5==5.10.1',  # to avoid PyQt5.sip being separated
                      # 'torch==0.4.0',  # can't install pytorch from pip, use conda
                      ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        # 'dev': ['tensorflow', 'ipdb', 'tensorboard', 'visdom', 'tensorboardX'],
        # 'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    package_data={  # Optional
        # 'miprometheus.config': ['default_config.yaml'],
        # we should list here some configs.yaml files that will be copied over with the data.
    '':[path.join(here,'miprometheus/problems/seq_to_seq/video_text_to_class/cog/cog_utils/roboto.ttf')]},

    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    entry_points={  # Optional
         'console_scripts': [
             'mip-grid-trainer-cpu=miprometheus.grid_workers.grid_trainer_cpu:main',
             'mip-grid-trainer-gpu=miprometheus.grid_workers.grid_trainer_gpu:main',
             'mip-grid-tester-cpu=miprometheus.grid_workers.grid_tester_cpu:main',
             'mip-grid-tester-gpu=miprometheus.grid_workers.grid_tester_gpu:main',
             'mip-grid-analyzer=miprometheus.grid_workers.grid_analyzer:main',
             'mip-index-splitter=miprometheus.helpers.index_splitter:main',
             'mip-offline-trainer=miprometheus.workers.offline_trainer:main',
             'mip-online-trainer=miprometheus.workers.online_trainer:main',
             'mip-tester=miprometheus.workers.tester:main',
         ],
     },

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'Documentation': 'https://mi-prometheus.readthedocs.io/en/latest/',
        'Source': 'https://github.com/IBM/mi-prometheus/',
        'Bug Reports': 'https://github.com/IBM/mi-prometheus/issues',
    },
)

