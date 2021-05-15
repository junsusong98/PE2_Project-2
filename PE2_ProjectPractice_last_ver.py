import xml.etree.ElementTree as et

def Polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    p = np.poly1d(coeffs)

    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot

    return results

# Parsing xml

xtree = et.parse('C:/Users/user/PycharmProjects/PE2_WorkingSP/P184640_D08_(0,2)_GORILLA5_DCM_LMZC.xml')
xroot = xtree.getroot()

# Extract tag data of 'TestSiteInfo'

element = xroot.findall('TestSiteInfo')

data = element[0].attrib

Batch = data['Batch']
DieColumn = data['DieColumn']
DieRow = data['DieRow']
Maskset = data['Maskset']
TestSite = data['TestSite']
Wafer = data['Wafer']

print('Batch : ' + Batch)
print()
print('DieColumn : ' + DieColumn)
print()
print('DieRow : ' + DieRow)
print()
print('Maskset : ' + Maskset)
print()
print('TestSite : ' + TestSite)
print()
print('Wafer : ' + Wafer)

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
L7 = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
IL7 = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
L7 = L7.text.split(",")
IL7 = IL7.text.split(",")
L7 = list(map(float, L7))
IL7 = list(map(float, IL7))

#plt.figure(1)
plt.subplot2grid((3,3), (0,0), rowspan = 1, colspan = 1)
plt.title('Transmission spectra - as measured')
for i in range (1, 7):
    L = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/L".format(i))
    IL = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/IL".format(i))
    Li = L.text.split(",")
    ILi = IL.text.split(",")
    Li = list(map(float, Li))
    ILi = list(map(float, ILi))
    DBias = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]".format(i))
    plt.plot(Li, ILi, '.', label = DBias.get('DCBias') + 'V')
plt.scatter(L7, IL7, s = 15, label = 'Reference', alpha = 0.05, facecolor = 'none', edgecolor = 'r')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')
plt.legend(loc = ('best'))

fp2 = np.polyfit(L7, IL7, 2)
f2 = np.poly1d(fp2)

fp3 = np.polyfit(L7, IL7, 3)
f3 = np.poly1d(fp3)

fp4 = np.polyfit(L7, IL7, 4)
f4 = np.poly1d(fp4)

fp5 = np.polyfit(L7, IL7, 5)
f5 = np.poly1d(fp5)

fp6 = np.polyfit(L7, IL7, 6)
f6 = np.poly1d(fp6)

fp7 = np.polyfit(L7, IL7, 7)
f7 = np.poly1d(fp7)

fp8 = np.polyfit(L7, IL7, 8)
f8 = np.poly1d(fp8)

degree = [2, 3, 4, 5, 6, 7, 8]
R = []
for i in range(2, 9):
    Ri = Polyfit(L7, IL7, i)
    R.append(Ri['determination'])
print(str(degree[R.index(max(R))]) + ':' + str(max(R)))

#plt.figure(2)
plt.subplot2grid((3,3), (0,1), rowspan = 1, colspan = 1)
plt.title('Reference fitting')
plt.plot(L7, IL7, '.', label = 'reference')
plt.plot(L7, f2(L7), label = '2nd order, R^2 = %0.8f' %R[0])
plt.plot(L7, f3(L7), label = '3rd order, R^2 = %0.8f' %R[1])
plt.plot(L7, f4(L7), label = '4th order, R^2 = %0.8f' %R[2])
plt.plot(L7, f5(L7), label = '5th order, R^2 = %0.8f' %R[3])
plt.plot(L7, f6(L7), label = '6th order, R^2 = %0.8f' %R[4])
plt.plot(L7, f7(L7), label = '7th order, R^2 = %0.8f' %R[5])
plt.plot(L7, f8(L7), label = '8th order, R^2 = %0.8f' %R[6])
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')
plt.legend(loc = ('best'))

#plt.figure(4)
plt.subplot2grid((3,3), (2,0), rowspan = 1, colspan = 1)
plt.title('IV - analysis')
plt.plot(vfloat, absi, 'bo')
plt.title('IV - analysis')
plt.xlabel('Voltage [V]')
plt.ylabel('Current[A]')
plt.yscale('log')

plt.subplot2grid((3,3), (0,2), rowspan = 1, colspan = 1)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')
plt.title('Transmission spectra - as processed')
for i in range(1, 7):
    L = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/L".format(i))
    IL = xtree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/IL".format(i))
    Li = L.text.split(",")
    ILi = IL.text.split(",")
    Li = list(map(float, Li))
    ILi = list(map(float, ILi))
    flati = ILi - f6(Li)
    plt.plot(Li, flati)

from lmfit import Model
from numpy import exp

def icurrent(v, s, t, p, c, a, b, r):
    return abs(s*(exp((v)/t) - 1) - p * a * (exp(b*(v + r)/2*t)-1)) + c

gmodel = Model(icurrent)
result = gmodel.fit(absi, v=vfloat, s = 1, t = 1, p = 1, c = 10/9, a = 1, b = 1, r = -1.25)


def LmfitR(y):

    yhat = result.best_fit
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results = ssreg / sstot
    return results

plt.subplot2grid((3,3), (2,1), colspan= 1 , rowspan= 1)
plt.plot(vfloat, absi, 'bo')
plt.plot(vfloat, result.best_fit, 'r-', label='best fit, R^2 = %f' %LmfitR(absi))
plt.legend(loc = 'best')
plt.text(-1,result.best_fit[4], str(result.best_fit[4]),
         color = 'r', horizontalalignment = 'center', verticalalignment = 'top' )
plt.text(1,result.best_fit[12], str(result.best_fit[12]),
         color = 'r', horizontalalignment = 'center', verticalalignment = 'top' )
plt.title('IV - fitting')
plt.xlabel('Voltage [V]')
plt.ylabel('Current[A]')
plt.yscale('log')

plt.show()


