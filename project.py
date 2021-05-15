import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from lmfit import Model
import datetime
from dateutil.parser import parse
import glob
import os
targetDir = r"C:\Users\junsu\PycharmProjects\PE02\week4\P184640\D07\20190715_190855"
file_list = os.listdir(targetDir)
xml_list = []
for file in file_list:
    if 'LMZ' in file:
        xml_list.append(file)


def TestSiteInfo(x,y): #Lot, Wafer,Maskset,TestSite,DieRow,DieColumn
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    a = tree.find("./TestSiteInfo")
    return (a.get(y))
# print(TestSiteInfo(xml_list[0],"DieRow"))
def Name(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    b= tree.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/DeviceInfo')
    return (b.get("Name"))

def Date(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    c= tree.find("./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo")
    w =str(parse(c.get("DateStamp")))
    return (w[0:4]+w[5:7]+w[8:10]+"_"+w[11:13]+w[14:16]+w[17:19])

def Wavelength(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    d = tree.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/DeviceInfo/DesignParameters/DesignParameter[2]')
    return d.text

def polyfitT(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results = ssreg / sstot
    return results

def Rsq_Ref(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    L7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    IL7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    L7 = L7.text.split(",")
    IL7 = IL7.text.split(",")
    L7 = list(map(float, L7))
    IL7 = list(map(float, IL7))
    Rsq_Ref = polyfitT(L7, IL7, 6)
    return Rsq_Ref

def transmission(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    IL7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    IL7 = IL7.text.split(",")
    IL7 = list(map(float, IL7))
    return max(IL7)

def Rsq_fit(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    b = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Voltage")
    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")
    x_2 = b.text.split(",")
    y_2 = c.text.split(",")
    x_list = list(map(float, x_2))
    y_list = list(map(float, y_2))
    y_list_1 = []
    for i in range(len(y_list)):
        g = abs(y_list[i])
        y_list_1.append(g)

    polyfiti = np.polyfit(x_list, y_list_1, 12)
    fiti = np.poly1d(polyfiti)

    def gaussian(x, q, w, alp):
        return abs(q * (exp(x / w) - 1)) + alp * fiti(x)

    gmodel = Model(gaussian)
    result = gmodel.fit(y_list_1, x=x_list, q=1, w=1, alp=1)

    yhat = result.best_fit
    ybar = np.sum(y_list_1) / len(y_list_1)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y_list_1 - ybar) ** 2)
    results = ssreg / sstot
    return results

def negative1(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")
    y_2 = c.text.split(",")
    y_list = list(map(float, y_2))
    y_list_1=[]
    for i in range(len(y_list)):
        g = abs(y_list[i])
        y_list_1.append(g)
    return y_list_1[4]
def positive1(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")
    y_2 = c.text.split(",")
    y_list = list(map(float, y_2))
    y_list_1=[]
    for i in range(len(y_list)):
        g = abs(y_list[i])
        y_list_1.append(g)
    return y_list_1[12]
def Errorcheck(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    L7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    IL7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    L7 = L7.text.split(",")
    IL7 = IL7.text.split(",")
    L7 = list(map(float, L7))
    IL7 = list(map(float, IL7))
    Rsq_Ref = polyfitT(L7, IL7, 6)
    if Rsq_Ref >= 0.996:
        return "No Error"
    else:
        return "Rsq_Ref Error"
def ErrorFlag(x):
    tree = ET.parse('./P184640/D07/20190715_190855/{}'.format(x))
    L7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    IL7 = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    L7 = L7.text.split(",")
    IL7 = IL7.text.split(",")
    L7 = list(map(float, L7))
    IL7 = list(map(float, IL7))
    Rsq_Ref = polyfitT(L7, IL7, 6)
    if Rsq_Ref >= 0.996:
        return 0
    else:
        return 1
