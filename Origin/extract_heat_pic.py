import numpy as np


def extract():
    name="data/Origin_Fermi_Qlearning2/heat_pic3_T/r=2.9_eta=0-1_gamma=0-1的热图数据_三位小数版本.csv"
    results = np.loadtxt(name, delimiter=",")
    print(results[87,53])

def find_zero():
    name="data/Origin_Fermi_Qlearning2/heat_pic3_T/r=2.9_eta=0-1_gamma=0-1的热图数据_三位小数版本.csv"
    results = np.loadtxt(name, delimiter=",")
    i=0
    for gamma in range(101):
        for eta in range(101):
            # if (77/78*eta/100+0.77-77/78)<=(gamma/100) and results[gamma,eta]<0.6 and gamma<=99 and eta>=15:
            if (eta>20 and eta<40 and gamma>80 and gamma<90 and results[gamma,eta]<0.6) or (eta>50 and eta<60 and gamma>84 and gamma<94 and results[gamma,eta]<0.6) or (eta>80 and eta<92 and gamma>80 and gamma<=99 and results[gamma,eta]<0.5):
                #存储gamma和eta到CSV文件中
                print(f"gamma={gamma/100},eta={eta/100},value={results[gamma,eta]}")
                with open("data/Origin_Fermi_Qlearning2/heat_pic3_T/zero.csv", "a") as f:
                    f.write(f"{gamma},{eta}\n")
                i+=1
    print(i)


#extract()
find_zero()