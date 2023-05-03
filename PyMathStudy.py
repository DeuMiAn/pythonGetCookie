import torch


# # 2x3 크기의 텐서를 생성합니다.
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# # dim=0으로 gather합니다.
# # 첫번째 인덱스는 [1, 2, 3]이고, 두번째 인덱스는 [4, 5, 6]입니다.
# # 따라서 idx가 0이면 [1, 2, 3], 1이면 [4, 5, 6]이 반환됩니다.
# # idx = torch.tensor([0, 1]).unsqueeze(1).repeat(1, 3)
# # print(idx)
# idx=[[1,0],[0,1]]
# idx=torch.tensor(idx)
# print(idx)
# result = x.gather(0, idx)
# print(result)

# import torch

# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# idx = torch.tensor([1, 2]).unsqueeze(0).repeat(2, 1)

# print(idx)
# # tensor([[1, 2],[1, 2]])

# result = x.gather(1, idx)
# print(result)
# # tensor([[2, 3], [5, 6]])

A = torch.Tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])

# torch.gather 함수를 써서 해보세요!

index=torch.arange(2).expand(2,2).reshape(2,1,2)
print(index)
# output = torch.gather(A,1,index).view(2,2)
# print(A[1][0][0])
# # output=A
# print(output)

# # 아래 코드는 수정하실 필요가 없습니다!
# if torch.all(output == torch.Tensor([[1, 4], [5, 8]])):
#     print("🎉🎉🎉 성공!!! 🎉🎉🎉")
# else:
#     print("🦆 다시 도전해봐요!")