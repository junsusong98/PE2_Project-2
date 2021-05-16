import numpy as np

def IV_data (root):
    V_data = root.iter('Voltage')
    I_data = root.iter('Current')

    v1 = [v.text for v in V_data]
    i1 = [i.text for i in I_data]

    v2 = v1[0].split(',')
    i2 = i1[0].split(',')

    voltage = list(map(float, v2))
    i3 = list(map(float, i2))

    current = []
    for i in range(0, len(i3)):
        current.append(abs(i3[i]))

    return voltage, current

def Wavelength_Sweep (root):
    return