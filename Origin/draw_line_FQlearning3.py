import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Origin.Function.draw_line_pic import draw_line_pic
from Origin.Function.draw_line2 import draw_line2
from Origin.Function.draw_value_line import draw_value_line
from Origin.Function.shot_pic import shot_pic,draw_all_shot_pic
from Origin.draw_line_FQlearning2 import com_zhuzhuang
L_num=200
#colors=['red','green','blue','black']
colors=[(217/255,82/255,82/255),(31/255,119/255,180/255),(120/255,122/255,192/255),(161/255,48/255,63/255),'gold','green']
labels=['D','C']
type_labels=['DD','CC','CDC','StickStrategy']
xticks=[-1,0, 10, 100, 1000, 10000, 100000]
r_xticks=[0,1,2,3,4,5]
fra_yticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95, 1.00]
# profite_yticks=[ 8,10,12,14,16,18,20,22]
profite_yticks=[ 0,2,4,6,8,10,12,14,16,18,20,22]
all_value_sum_yticks=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
log_start_points = [2,4,6,8,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,8000,10000]


if __name__ == '__main__':
    r=3.5
    loop_num=10
    Origin_Fermi_Qlearning3="Origin_Fermi_Qlearning3"
    name=["D_fra","C_fra"]
    for i in range(35,36):
        r=i/10
        draw_all_shot_pic(r,epoches=20000,L_num=200,count=0,updateMethod=Origin_Fermi_Qlearning3,generated="generated1")
    #折线图随时间
    #draw_line1(loop_num, name, r, Origin_Fermi_Qlearning3,epoches=100000,L_num=200)
    #draw_all_shot_pic(r,epoches=10000,L_num=200,count=0,updateMethod=Origin_Fermi_Qlearning3,generated="generated1")
    #draw_line1(loop_num, name, r, Origin_selfQlearning,epoches=10000)

    #折线图随r
    #draw_line2(50, 10, Origin_Fermi_Qlearning3,epoches=20000)
    #draw_line2(51, 10, Origin_selfQlearning,epoches=10000)

    #all_value折线图
    #draw_all_value_line(loop_num,'all_value',r,ylim=(0,22))

    #value折线图
    #draw_value_line(loop_num,['D_value','C_value'],r,Origin_Fermi_Qlearning2,ylim=(0,14),epoches=50000)
    #draw_value_line(loop_num,['D_value','C_value'],r,Origin_selfQlearning,ylim=(0,14),epoches=10000)

    #Qtable
    #draw_line_four_type(5, ["Q_D_DD", "Q_D_DC", "Q_D_CD", "Q_D_CC"], na='QtableD',epoches=50000, r=r,updateMethod=Origin_Fermi_Qlearning2, labels=['D_DD', 'D_CD', 'D_DC', 'D_CC'], ylim=(0, 40),yticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], ylabel='Fractions')
    #draw_line_four_type(5, ["Q_C_DD", "Q_C_DC", "Q_C_CD", "Q_C_CC"],na='QtableD',epoches=50000, r=r, updateMethod=Origin_Fermi_Qlearning2,labels=['C_DD','C_CD','C_DC','C_CC'], ylim=(0, 40), yticks=[ 0,5,10,15,20,25,30,35,40,45,50,55,60], ylabel='Fractions')
    #draw_line_four_type(loop_num, ["Q_D_DD", "Q_D_DC", "Q_D_CD", "Q_D_CC"], na='QtableD',epoches=10000, r=r,updateMethod=Origin_selfQlearning, labels=['D_DD', 'D_CD', 'D_DC', 'D_CC'], ylim=(0, 200),yticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,200], ylabel='Fractions')
    #draw_line_four_type(loop_num, ["Q_C_DD", "Q_C_DC", "Q_C_CD", "Q_C_CC"],na='QtableC',epoches=10000, r=r, updateMethod=Origin_selfQlearning,labels=['C_DD','C_CD','C_DC','C_CC'], ylim=(0, 200),yticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,200], ylabel='Fractions')
    #zhuzhuangtu(5,["Q_D_DD", "Q_D_DC", "Q_D_CD", "Q_D_CC"],r,Origin_Fermi_Qlearning2,epoches=50000,L_num=200)
    #zhuzhuangtu(5, ["Q_C_DD", "Q_C_DC", "Q_C_CD", "Q_C_CC"], r, Origin_Fermi_Qlearning2, epoches=50000, L_num=200)
    #com_zhuzhuang(["Q_D_DD", "Q_D_DC", "Q_D_CD", "Q_D_CC"], ["Q_C_DD", "Q_C_DC", "Q_C_CD", "Q_C_CC"], r, Origin_Fermi_Qlearning2, epoches=50000, L_num=200)
