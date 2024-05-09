import torch

if __name__ == '__main__':
    L_num = 200
    type_t_matrix = torch.randint(0, 2, (L_num, L_num)).to(torch.float16).to("cpu")
    changetype = torch.where(type_t_matrix == 0,1,0)
    changetype[:int(L_num / 2), :] = 0
    changetype=changetype.view(-1)
    indices = torch.nonzero(changetype == 1)
    print(indices)
