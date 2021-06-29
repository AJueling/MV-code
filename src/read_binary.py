# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 08:53:13 2017

@author: andre
"""

import numpy as np



def read_binary_2D(inputfilename, nx, ny, nrec):
    """
    reads binary files with real/float (4 byte) big endian numbers
    where nrec is the record number in binary file
    returns a 2D field of dimension (nx,ny)
    """
    f = open(inputfilename, 'rb')
    f.seek(4*(nrec-1)*nx*ny)
    field = np.fromfile(f, dtype='>f4', count=nx*ny)
    field = np.reshape(field, (ny, nx)).T
    f.close()
    return field


def read_binary_2D_double(inputfilename, nx, ny, nrec):
    """
    reads binary files with double (8 byte) big endian numbers
    where nrec is the record number in binary file
    returns a 2D field of dimension (nx,ny)
    """
    f = open(inputfilename, 'rb')
    f.seek(8*(nrec-1)*nx*ny)
    field = np.fromfile(f, dtype='>f8', count=nx*ny)
    field = np.reshape(field, (ny, nx)).T
    f.close()
    return field


def read_binary_2D_int(inputfilename, nx, ny, nrec):
    """
    reads binary files with integer big endian numbers
    where nrec is the record number in binary file
    returns a 2D field of dimension (nx,ny)
    """
    f = open(inputfilename, 'rb')
    f.seek(8*(nrec-1)*nx*ny)
    field = np.fromfile(f, dtype='>i', count=nx*ny)
    field = np.reshape(field, (ny, nx)).T
    f.close()
    return field


def read_binary_3D(filename, nx, ny, nz, nrec=1):
    """
    returns complete 3D field of dimension (nx,ny,nz)
    """
    field = np.zeros((nx, ny, nz))
    for k in np.arange(nrec, nrec+nz):
        field[:, :, k-nrec] = read_binary_2D(filename, nx, ny, k)
    return field


def write_binary_2D(outputfilename, field, nx, ny):
    """
    writes a 2D numpy array to a binray file that can be read with the above functions
    """
    field = np.reshape(np.array(field, dtype='>f4').T, nx*ny)
    f = open(outputfilename, 'wb')
    field.tofile(f)