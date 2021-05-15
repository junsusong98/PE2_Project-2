import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from numpy import exp


xtree = et.parse('C:/Users/user/PycharmProjects/PE2_WorkingSP/data/P184640/D07/20190715_190855/P184640_D07_(0,0)_GORILLA5_DCM_LMZC.xml')
xroot = xtree.getroot()

cols = ['Lot', 'Wafer', 'Mask', 'TestSite', 'Name', 'Date',
        #        'Script ID', 'Script Version',
        'Script Owner', 'Operator', 'Row', 'Column',
        'ErrorFlag', 'Error description',
        'Analysis Wavelength', 'Rsq of Ref.spectrum (6th)',
        'Max transmission of Ref. spec. (dB)',
        'Rsq of IV', 'I at -1V [A]', 'I at 1V [A]']

L7 = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
IL7 = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
L7 = L7.text.split(",")
IL7 = IL7.text.split(",")
L7 = list(map(float, L7))
IL7 = list(map(float, IL7))

Vdata = xroot.iter('Voltage')
Idata = xroot.iter('Current')
voltage = [v.text for v in Vdata]
current = [i.text for i in Idata]
vtext = voltage[0].split(',')
itext = current[0].split(',')
vfloat = list(map(float, vtext))
ifloat = list(map(float, itext))
absi = []
for i in range(0, len(ifloat)):
    absi.append(abs(ifloat[i]))

polyfiti = np.polyfit(vfloat, absi, 12)
fiti = np.poly1d(polyfiti)


def IVfitting(x, q, w, alp):
    return abs(q * (exp(x / w) - 1)) + alp * fiti(x)


gmodel = Model(IVfitting)
result = gmodel.fit(absi, x=vfloat, q=1, w=1, alp=1)


def LmfitR(y):
    yhat = result.best_fit
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results = ssreg / sstot
    return results


subrows = []
rows = []


def check_tagdata(nodename):
    node = xroot.iter(nodename)
    node = list(node)
    data = node[0].attrib
    dict = {}
    for i in range(0, len(data)):
        key = list(data.keys())
        value = list(data.values())
        dict[key[i]] = value[i]
    subrows.append(dict)


def time_extract(nodename):
    dict = {}
    node = xroot.iter(nodename)
    node = list(node)
    data = node[0].attrib
    for i in range(0, len(data)):
        key = list(data.keys())
        value = list(data.values())
        dict[key[i]] = value[i]

    date = str(dict['DateStamp'])

    DataList = date.split()

    month = DataList[1]
    day = DataList[2]
    time = DataList[3]
    year = DataList[4]

    if month == 'Jan':
        DataList.remove('Jan')
        DataList.insert(1, '01')
    elif month == 'Feb':
        DataList.remove('Feb')
        DataList.insert(1, '02')
    elif month == 'Mar':
        DataList.remove('Mar')
        DataList.insert(1, '03')
    elif month == 'Apr':
        DataList.remove('Apr')
        DataList.insert(1, '04')
    elif month == 'May':
        DataList.remove('May')
        DataList.insert(1, '05')
    elif month == 'Jun':
        DataList.remove('Jun')
        DataList.insert(1, '06')
    elif month == 'Jul':
        DataList.remove('Jul')
        DataList.insert(1, '07')
    elif month == 'Aug':
        DataList.remove('Aug')
        DataList.insert(1, '08')
    elif month == 'Sep':
        DataList.remove('Sep')
        DataList.insert(1, '09')
    elif month == 'Oct':
        DataList.remove('Oct')
        DataList.insert(1, 10)
    elif month == 'Nov':
        DataList.remove('Nov')
        DataList.insert(1, '11')
    else:
        DataList.remove('Dec')
        DataList.insert(1, '12')

    timeset = time.split(':')
    timestr = str(timeset[0]) + str(timeset[1]) + str(timeset[2])

    dateset = str(year) + str(DataList[1]) + str(day) + '_' + timestr

    subrows.append(dateset)


def Polyfit(x, y, degree):
    coeffs = np.polyfit(x, y, degree)

    p = np.poly1d(coeffs)

    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results = ssreg / sstot

    return results


check_tagdata('TestSiteInfo')

check_tagdata('DeviceInfo')

time_extract('PortCombo')

check_tagdata('OIOMeasurement')



analysis_wavelength = xroot.iter('DesignParameter')
Analysis_Wavelength = [w.text for w in analysis_wavelength]

Rsq_Ref = Polyfit(L7, IL7, 6)

if 0.95 < Rsq_Ref and 0.95 < LmfitR(absi):
    e_num = 0
elif 0.95 >= Rsq_Ref and 0.95 < LmfitR(absi):
    e_num = 1
elif 0.95 < Rsq_Ref and 0.95 >= LmfitR(absi):
    e_num = 2
else: # 0.95 >= Rsq_Ref and 0.95 >= LmfitR(absi)
    e_num = 3

if e_num == 0:
    des = '0'
elif e_num == 1:
    des = 'Ref. spec. Error'
elif e_num == 2:
    des = 'IV Error'
else:
    des = 'Ref. spec. and IV Error'

rows.append({'Lot': subrows[0]['Batch'],
             'Wafer': subrows[0]['Wafer'],
             'Mask': subrows[0]['Maskset'],
             'TestSite': subrows[0]['TestSite'],
             'Name': subrows[1]['Name'],
             'Date': subrows[2],
             'Script Owner': 'B1',
             'Operator': subrows[3]['Operator'],
             'Row': subrows[0]['DieRow'],
             'Column': subrows[0]['DieColumn'],
             'ErrorFlag': str(e_num),
             'Error description': des,
             'Analysis Wavelength': Analysis_Wavelength[1],
             'Rsq of Ref.spectrum (6th)': Rsq_Ref,
             'Max transmission of Ref. spec. (dB)': '%0.2f' % max(IL7),
             'Rsq of IV': str(LmfitR(absi)),
             'I at -1V [A]': str(ifloat[4]),
             'I at 1V [A]': str(ifloat[12])
             })

df = pd.DataFrame(rows, columns = cols)

df.to_csv('AnalysisResult_B1.csv')

train = pd.read_csv('AnalysisResult_B1.csv')

from dateutil.parser import parse

