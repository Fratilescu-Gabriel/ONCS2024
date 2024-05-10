# ONCS2024


## Library list

__pytorch x__ - \
__TensorFlow x__ - \
__numpy__ - https://numpy.org/doc/1.26/  \
__SciPy__ - https://docs.scipy.org/doc/scipy/  \
__pandas__ - https://pandas.pydata.org/docs/  \
__matplotlib__ - https://matplotlib.org/stable/index.html  \
__~plotly__ - Plotting X\
__??pyVista__ - \
__????? Tkinter or PyQT5/PyQT6__ - GUI \
__?geoPandas__ - geospacial data processing \
__?xarray__ - (https://docs.xarray.dev/en/stable/) \
__??PyKE__ - \
__#Astropy__ - https://docs.astropy.org/en/stable/ \
__#sunpy__ - https://docs.sunpy.org/en/stable/ \
__pandas3D__- https://docs.panda3d.org/1.10/python/index \


.LBL, .IMG files -> PNG si matplotlib, opencv / to binary

TO BINARY:
def read_lbl_file(filename):
    with open(filename, 'rb') as file:
        # Read the file based on your known structure
        # For example, read header, metadata, then pixel data
        header = file.read(header_length)  # Adjust `header_length` appropriately
        # Depending on the header info, you might read pixel data differently
        pixels = file.read()  # This might need more structure
        return pixels

pixels = read_lbl_file('yourfile.lbl')

https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html