import distutils
import os
import subprocess
import sys
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

extensions = [
    Extension("coder.baseline_decoder", ["coder/baseline_decoder.pyx"]),
    Extension("coder.baseline_encoder", ["coder/baseline_encoder.pyx"]),
]


class InstallRequirements(distutils.cmd.Command):

    description = "Install requirements"
    user_options = []

    def initialize_options(self):
        self.cwd = None

    def finalize_options(self):
        self.cwd = os.getcwd()

    def run(self):

        res = subprocess.run(
            [sys.executable, '-m', 'conda', 'install', '-y', 'pytorch=0.4', '-c', 'pytorch'])
        assert res.returncode == 0, "Error"

        res = subprocess.run(
            [sys.executable, '-m', 'conda', 'install', '-y', 'opencv'])
        assert res.returncode == 0, "Error"

        res = subprocess.run(
            [sys.executable, '-m', 'conda', 'install', '-y', '-c', 'conda-forge', 'scikit-image'])
        assert res.returncode == 0, "Error"

        res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        assert res.returncode == 0, "Error"


setup(
    name='Structured OCR',
    description='A line localization and text recognition tools using Deep Learning with PyTorch',
    author='BELOS Thomas',
    url='https://github.com/belosthomas/socr',
    cmdclass={
        'build_ext': build_ext,
        'install_requirements': InstallRequirements,
    },
    ext_modules=cythonize(extensions),
    packages=['socr', 'scribbler']
)
