from Function.shot_pic import shot_pic,draw_all_shot_pic


if __name__ == '__main__':
    r=3.5
    loop_num=10
    Origin_Qlearning="Origin_Qlearning"
    name=["D_fra","C_fra"]
    for i in [47]:
        r=i/10
        draw_all_shot_pic(r,epoches=10000,L_num=200,count=0,updateMethod=Origin_Qlearning,generated="generated3")
