import torch

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def split_four_policy_type( Q_matrix):
    CC = torch.where((Q_matrix[:, 1, 1] > Q_matrix[:, 1, 0]) & (
                Q_matrix[:, 0, 0] < Q_matrix[:, 0, 1]), torch.tensor(1), torch.tensor(0))
    # CC=(Q_matrix[:,1,1]>Q_matrix[:,1,0])&(Q_matrix[:,0,0]<Q_matrix[:,0,1])
    print("CC:")
    print(CC)
    DD = torch.where((Q_matrix[:, 0, 0] > Q_matrix[:, 0, 1]) & (
                Q_matrix[:, 1, 1] < Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0))
    # DD=(Q_matrix[:,0,0]>Q_matrix[:,0,1])&(Q_matrix[:,1,0]>Q_matrix[:,1,1])
    print("DD:")
    print(DD)
    CDC = torch.where((Q_matrix[:, 0, 0] <= Q_matrix[:, 0, 1]) & (
                Q_matrix[:, 1, 1] <= Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0))
    # CDC=(Q_matrix[:,0,1]>Q_matrix[:,0,0])&(Q_matrix[:,1,1]>Q_matrix[:,1,0])
    print("CDC:")
    print(CDC)
    StickStrategy=torch.where((Q_matrix[:,0,0]>Q_matrix[:,0,1])&(Q_matrix[:,1,1]>Q_matrix[:,1,0]),torch.tensor(1),torch.tensor(0))
    # StickStrategy = (Q_matrix[:, 0, 0] > Q_matrix[:, 0, 1]) & (Q_matrix[:, 1, 1] < Q_matrix[:, 1, 0])
    print("StickStrategy:")
    print(StickStrategy)


if __name__ == '__main__':
    Q_table=torch.randint(0,10,(9,2,2)).to(device)
    Q_table[0]=torch.tensor([[0,0],[0,0]])
    print(Q_table)
    split_four_policy_type(Q_table)
