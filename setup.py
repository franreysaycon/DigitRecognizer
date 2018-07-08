from distutils.core import setup
import py2exe

setup(name="DigitRecognizer ver1.1",
      console=["DigitRecognizer.py"],    # put the name of your main script here
      options = {"py2exe": {"packages": ["encodings"]}}
)
