# multinest Copyright (C) 2010 Farhan Feroz & Mike Hobson

import os
import re
import sys
import setuptools
from setuptools import setup, Extension

from numpy import get_include


__version__ = '0.0.0'

#srcdir = 'src_c/'          # C-code source folder
srcdir = './'          # C-code source folder
incdir = f'{srcdir}include/'  # Include folder with header files

cfiles = os.listdir(srcdir)
cfiles = list(filter(lambda x:     re.search('.+[.]c$',     x), cfiles))
cfiles = list(filter(lambda x: not re.search('[.#].+[.]c$', x), cfiles))

cfiles = [
    'random_ns.c',
    'utils.c',
    'kmeans.c',
#    'xmeans.c',
    'multinest.c',
    ]

cfiles = [f'{srcdir}{cfile}' for cfile in cfiles]

inc = [
    get_include(),
    incdir,
    ]

eca = ['-ffast-math']
ela = []

extensions = []
#for cfile in cfiles:
e = Extension('lib.multinest',
        #sources=[f'{srcdir}{cfile}'],
        sources=cfiles,
        include_dirs=inc,
        libraries=['lapack'],
        extra_compile_args=eca,
        extra_link_args=ela)
extensions.append(e)

#with open('README.md', 'r') as f:
#    readme = f.read()

setup(name = "multinest",
      version = __version__,
      author = "Patricio Cubillos",
      author_email = "patricio.cubillos@oeaw.ac.at",
      #url = "https://github.com/pcubillos/multinest",
      packages = setuptools.find_packages(),
      #install_requires = ['numpy>=1.8.1',
      #                    'scipy>=0.13.3',
      #                    'matplotlib>=1.3.1',
      #                    'sympy>=0.7.6',
      #                    'mc3>=3.0.0',
      #                   ],
      #tests_require = [
      #    'pytest>=3.9',
      #    'scipy>=1.4.1',
      #    ],
      #license = "GNU GPLv2",
      description = "C implementation of multinest.",
      #long_description=readme,
      long_description_content_type="text/markdown",
      include_dirs = inc,
      #entry_points={"console_scripts": ['mnest = multinest.__main__:main']},
      ext_modules = extensions)
