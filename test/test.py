import torch
import articulate as art


# Quaternion: wxyz
# pose_q = torch.tensor([1, 0, 0, 0.]).expand(24, 4)
# pose_R = art.math.quaternion_to_rotation_matrix(pose_q).view(1, 24, 3, 3)
# m = art.ParametricModel('models/SMPL_male.pkl')
# m.view_motion([pose_R])


pose_q = torch.tensor([0, 1, 0, 0.]).expand(24, 4)
pose_R = art.math.quaternion_to_rotation_matrix(pose_q).view(1, 24, 3, 3)
m = art.ParametricModel('models/SMPL_male.pkl')
m.view_motion([pose_R])


# pose_q = torch.rand(24, 4)
# pose_q = pose_q / pose_q.norm(dim=1, keepdim=True)
# pose_R = art.math.quaternion_to_rotation_matrix(pose_q).view(1, 24, 3, 3)
# m = art.ParametricModel('models/SMPL_male.pkl')
# m.view_motion([pose_R])


# seq = 10
# data = torch.load('dataset_oppo/AMASS/pose.pt')
# pose_R = art.math.axis_angle_to_rotation_matrix(data[seq]).view(-1, 24, 3, 3)
# m = art.ParametricModel('models/SMPL_male.pkl')
# m.view_motion([pose_R[111]])


def f(x):
    return torch.norm(x)


# save = []
# x = torch.rand(24, 4)
# x.requires_grad_(True)
# for epoch in range(10000):
#     distance = f(x)
#     distance.backward()
#     print(distance)
#     with torch.no_grad():
#         x.data -= x.grad * 0.0001
#         x.grad[:] = 0
#         x.data = x.data / x.data.norm(dim=1, keepdim=True)
#         save.append(x.detach().clone())

# torch.stack(save)
