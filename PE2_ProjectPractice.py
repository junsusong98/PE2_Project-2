import xml.etree.ElementTree as et

# Parsing xml

xtree = et.parse('C:/Users/user/PycharmProjects/PE2_WorkingSP/P184640_D08_(0,2)_GORILLA5_DCM_LMZC.xml')
xroot = xtree.getroot()

# Extract tag data of 'TestSiteInfo'

def tagdata(nodename):
    nodedata = xroot.findall(nodename)
    node = list(nodedata)
    data = node[0].attrib
    dict = {}
    for i in range(0, len(data)):
        key = list(data.keys())
        value = list(data.values())
        dict[key[i]] = value[i]

    print(dict)

tagdata('TestSiteInfo')

# Extract and plot voltage - current data

Vdata = xroot.iter('Voltage')
Idata = xroot.iter('Current')

voltage = [v.text for v in Vdata]
current = [i.text for i in Idata]

vtext = voltage[0].split(',')
itext = current[0].split(',')

vfloat = []
ifloat = []

for i in range(0, len(vtext)):
    vfloat.append(float(vtext[i]))

for i in range(0, len(itext)):
    ifloat.append(float(itext[i]))

import matplotlib.pyplot as plt

import numpy as np

absi = []

for i in range(0, len(ifloat)):
    absi.append(abs(ifloat[i]))

logi = []
for i in range(0, len(itext)):
    logscale = (lambda c: np.log(c))(absi[i])
    logi.append(logscale)

# Extract and plot WavelengthSweep data

for i in range(1, 7):
    Li = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/L".format(i))
    ILi = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/IL".format(i))
    Bias = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]".format(i))
    bias = list(Bias)
    dcbias = bias[0].attrib
    DCBias = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]".format(i))
    Li = list(map(float, Li))
    ILi = list(map(float, ILi))
    plt.plot(Li, ILi, ',', label = DCBias)

plt.subplot2grid((2,3), (0,0), rowspan = 1, colspan = 1)
plt.title('Transmission spectra - as measured')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')
plt.legend(loc = (1.0, 0))

L7 = xroot.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/L".format(6))
IL7 = xroot.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/IL".format(6))

L7 = list(map(float, L7))
IL7 = list(map(float, IL7))


def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    p = np.poly1d(coeffs)

    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot

    return results

plt.subplot2grid((2,3), (0,1), rowspan = 1, colspan = 1)
plt.title('fitting')
plt.plot(L7, IL7, '.', label = 'reference')
for i in range(2, 9):
    fp = np.polyfit(L7, IL7, i)
    f = np.poly1d(fp)
    plt.plot(L7, f(L7), label='{}nd order').format(i)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')
plt.legend(loc = (1.0, 0))

fp6 = np.polyfit(L7, IL7, 6)
f6 = np.poly1d(fp6)

plt.subplot2grid((2,3), (1,0), rowspan = 1, colspan = 1)
plt.plot(vfloat, absi, 'bo')
plt.title('IV - analysis')
plt.xlabel('Voltage [V]')
plt.ylabel('Current[A]')
plt.yscale('log')

plt.subplot2grid((2,3), (0,2), rowspan = 1, colspan = 1)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')
plt.title('Transmission spectra - as processed')
for i in range(0,6):
    Li = xroot.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/L".format(i))
    ILi = xroot.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/IL".format(i))
    Li = list(map(float, Li))
    ILi = list(map(float, ILi))
    flat = ILi - f6(Li)
    plt.plot(Li, flat)




#s = 10**(-16)
#v = -1
#fitting = abs(s*(exp(v/0.026)-1))
#metadata = absi[4]

from lmfit import Model
from numpy import exp

def gaussian(v, s, p, t):
    return abs(s * (exp(v / t) - 1)) + p

gmodel = Model(gaussian)
result = gmodel.fit(absi, v = vfloat, s = 1, p = 1, t = 1)

plt.figure(3)
plt.plot(vfloat, absi, 'bo')
plt.plot(vfloat, result.best_fit, 'r-', label='best fit')
plt.legend(loc = 'best')
plt.title('IV - fitting')
plt.xlabel('Voltage [V]')
plt.ylabel('Current[A]')
plt.yscale('log')

plt.show()