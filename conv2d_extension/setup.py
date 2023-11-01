from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='conv2d_cudnn',
      ext_modules=[CppExtension('fastconv2d', ['conv2d_backward.cpp'])],
      cmdclass={'build_ext': BuildExtension})
