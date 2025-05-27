from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

# Find the pybind11 include directory
import pybind11
PYBIND11_INCLUDE = pybind11.get_include()

# Collect source files
source_files = ['src/hforest.cpp']

# Define the extension module
ext_modules = [
    Extension(
        'hforest',
        source_files,
        include_dirs=[
            'src',  # Add src directory to include path
            PYBIND11_INCLUDE
        ],
        libraries=['gomp'],
        language='c++'
    ),
]

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

# A custom build extension for adding compiler-specific options.
class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        if ct == 'unix':
            opts.append('-std=c++14')  # Use C++14
            opts.append('-fopenmp')    # Add OpenMP support
            opts.append('-march=native')  # Enable all available CPU instruction sets
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
            
            # Control NDEBUG based on DEBUG environment variable
            if os.environ.get('DEBUG'):
                # Debug mode: Disable NDEBUG to enable asserts
                if not hasattr(ext, 'undef_macros'):
                    ext.undef_macros = []
                ext.undef_macros.append('NDEBUG')
            else:
                # Release mode: Explicitly define NDEBUG
                ext.define_macros.append(('NDEBUG', '1'))
        
        build_ext.build_extensions(self)

setup(
    name='hforest',
    version='0.1.0',
    author='Yasunobu Imamura (COLUN)',
    author_email='imamura.kit@gmail.com',
    description='Hilbert Forest: A spatial indexing library using Hilbert curves',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.6',
)