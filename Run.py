import project as ifm
import numpy as np
import pandas as pd
import os
targetDir = r"C:\Users\junsu\PycharmProjects\PE02\week4\P184640\D07\20190715_190855"
file_list = os.listdir(targetDir)
xml_list = []
for file in file_list:
    if 'LMZ' in file:
        xml_list.append(file)

b=[]
for i in range(0,14):
     a = [ifm.TestSiteInfo(xml_list[i],"Batch"),
          ifm.TestSiteInfo(xml_list[i],"Wafer"),
          ifm.TestSiteInfo(xml_list[i],"Maskset"),
          ifm.TestSiteInfo(xml_list[i],"TestSite"),
          ifm.Name(xml_list[i]),
          ifm.Date(xml_list[i]),
          "process LMZ",
          "0.1",
          "B1",
          "B1 team member",
          ifm.TestSiteInfo(xml_list[i],"DieRow"),
          ifm.TestSiteInfo(xml_list[i],"DieColumn"),
          ifm.ErrorFlag(xml_list[i]),
          ifm.Errorcheck(xml_list[i]),
          ifm.Wavelength(xml_list[i]),
          ifm.Rsq_Ref(xml_list[i]),
          ifm.transmission(xml_list[i]),
          ifm.Rsq_fit(xml_list[i]),
          ifm.negative1(xml_list[i]),
          ifm.positive1(xml_list[i])]
     b.append(a)



df = pd.DataFrame(np.array(b),columns=['Lot','Wafer','Mask',
                                       'TestSite','Name','Date','Scrip ID','Script Version',
                                       "Script Owner","Operator","Row","Column"
                                        ,"ErrorFlag","Error description","Analysis Wavelengh",
                                       "Rsq of Ref.spectrum(6th)","Max transmission of Ref spec(dB)",
                                       "Rsq of IV","I at -1V[A]","I at 1V[A]"])
print(df)
df.to_csv("pandas.csv",mode="w")
detaset = pd.read_csv("pandas.csv")
print(detaset)
