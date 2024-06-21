from Origin.Function.read_data import read_data
from Origin.Function.draw_line_pic import draw_line_pic
import numpy as np

xticks=[-1,0, 10, 100, 1000, 10000, 100000]
r_xticks=[0,1,2,3,4,5]
fra_yticks=[0,0.2,0.4,0.6,0.8,1]

def draw_line2(loop_num1,loop_num2,updateMethod,epoches=10000,L_num=200):
    D_Final_fra = np.array([])
    C_Final_fra = np.array([])
    r = 0
    for i in range(loop_num1):
        D_Loop_fra=0
        C_Loop_fra=0
        for count in range(loop_num2):
            C_Y=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),'C_fra', 'C_fra', r/10, epoches,L_num, str(count)))
            D_Y=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),'D_fra', 'D_fra', r/10, epoches,L_num, str(count)))
            D_Loop_fra = D_Loop_fra+D_Y[-1]
            C_Loop_fra = C_Loop_fra+C_Y[-1]
        D_Final_fra = np.append(D_Final_fra, D_Loop_fra)
        C_Final_fra = np.append(C_Final_fra, C_Loop_fra)
        r = r + 1
    D_Final_fra = D_Final_fra / loop_num2
    C_Final_fra = C_Final_fra / loop_num2

    draw_line_pic( D_Final_fra, C_Final_fra, r_xticks, fra_yticks,r='0-5',updateMethod=updateMethod, epoches=epoches, type="line2", ylabel='Fractions', xlable='r')