from distutils.core import setup

setup(
    name="NameGenderClassifier",
    version="0.1",

    author="João Paulo Reis Alvarenga",
    author_email="joaopaulo.reisalvarenga@gmail.com",

    description="A gender classifier based on first names",
    long_description=open("README.md").read(),

    platforms='Linux',
    license='LICENSE',
    classifiers=[
        'Development Status :: 0 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Natural Language ::Portuguese (Brazilian)',
        'Operating System :: POSIX :: Linux',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
    ],

    keywords='classifier gender name nome classificador sexo genero',
    packages=['genderclassifier'],
    install_requires=['numpy', 'scipy', 'keras', 'tensorflow', 'pandas']
)