from bs4 import BeautifulSoup
from dateutil.parser import parse
import pandas as pd
import numpy as np

def to_csv ():
    columns = ['Lot', 'Wafer', 'Mask', 'TesSite', 'Name', 'Date',
               'Script ID', 'Script Version', 'Script Owner',
               'Operator', 'Row', 'Column',
               'ErrorFlag', 'Error description',
               'Analysis Wavelength',
               'Rsq of Ref.spectrum(Nth)',
               'Max transmission of Ref.spec.(dB)',
               'Rsq of IV',
               'I at -1V [A]', 'I at 1V [A]']

    fp = open("C:/Users/JaeUng/Desktop/P184640_D08_(0,2)_GORILLA5_DCM_LMZC.xml", "r")

    soup = BeautifulSoup(fp, "html.parser")

    Lot = soup.select('testsiteinfo')[0]['batch']
    Wafer = soup.select('testsiteinfo')[0]['wafer']
    Maskset = soup.select('testsiteinfo')[0]['maskset']
    Testsite = soup.select('testsiteinfo')[0]['testsite']
    Name = soup.select('modulator')[0]['name']
    date = soup.select('oiomeasurement')[0]['creationdate']
    Date = parse(date).strftime('%Y%m%d_%H%M%S')
    Script_ID = 'process LMZ'
    Script_ver = 0.1
    Script_owner = 'A02'
    Operator = 'JU.JEONG'
    Row = soup.select('testsiteinfo')[0]['dierow']
    Column = soup.select('testsiteinfo')[0]['diecolumn']