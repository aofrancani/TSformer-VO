import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from datasets.utils import rotation_to_euler
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


class KITTI(torch.utils.data.Dataset):
    """
    Dataloader for KITTI Visual Odometry Dataset
        http://www.cvlibs.net/datasets/kitti/eval_odometry.php

    Arguments:
        data_path {str}: path to data sequences
        gt_path {str}: path to poses
    """

    def __init__(self,
                 data_path=r"data/sequences_jpg",
                 gt_path=r"data/poses",
                 camera_id="2",
                 sequences=["00", "02", "08", "09"],
                 window_size=3,
                 overlap=1,
                 read_poses=True,
                 transform=None,
                 ):


        self.data_path = data_path
        self.gt_path = gt_path
        self.camera_id = camera_id
        self.frame_id = 0
        self.read_poses = read_poses
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform

        # KITTI normalization
        self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        # define sequence for training, test and val
        self.sequences = sequences

        # read frames list and ground truths
        frames, seqs = self.read_frames()
        gt = self.read_gt()

        # create dataframe with frames and ground truths
        data = pd.DataFrame({"gt": gt})
        data = data["gt"].apply(pd.Series)
        data["frames"] = frames
        data["sequence"] = seqs
        self.data = data
        self.windowed_data = self.create_windowed_dataframe(data)

    def __len__(self):
        return len(self.windowed_data["w_idx"].unique())

    def __getitem__(self, idx):
        """
        Returns:
            frame {ndarray}: image frame at index self.frame_id
            pose {list}: list containing the ground truth pose [x, y, z]
            frame_id {int}: integer representing the frame index
        """
        # get data of corresponding window index
        data = self.windowed_data.loc[self.windowed_data["w_idx"] == idx, :]

        # Read frames as grayscale
        frames = data["frames"].values
        imgs = []
        for fname in frames:
            img = Image.open(fname).convert('RGB')
            # pre processing
            img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.asarray(imgs)
        # T C H W -> C T H W.
        imgs = imgs.transpose(1, 0, 2, 3)

        # Read ground truth [window_size-1 x 6]
        gt_poses = data.loc[:, [i for i in range(12)]].values
        y = []
        for gt_idx, gt in enumerate(gt_poses):

            # homogeneous pose matrix [4 x 4]
            pose = np.vstack([np.reshape(gt, (3, 4)), [[0., 0., 0., 1.]]])

            # compute relative pose from frame1 to frame2
            if gt_idx > 0:
                pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose)
                R = pose_wrt_prev[:3, :3]
                t = pose_wrt_prev[:3, 3]

                # Euler parameterization (rotations as Euler angles)
                angles = rotation_to_euler(R, seq='zyx')

                # normalization
                angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
                t = (np.asarray(t) - self.mean_t) / self.std_t

                # concatenate angles and translation
                y.append(list(angles) + list(t))

            pose_prev = pose

        y = np.asarray(y)  # discard first value
        y = y.flatten()

        return imgs, y

    def read_intrinsics_param(self):
        """
        Reads camera intrinsics parameters

        Returns:
            cam_params {dict}: dictionary with focal lenght and principal point
        """
        calib_file = os.path.join(self.data_path, self.sequence, "calib.txt")
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            line = lines[int(self.camera_id)].strip().split()
            [fx, cx, fy, cy] = [float(line[1]), float(line[3]), float(line[6]), float(line[7])]

            # focal length of camera
            self.cam_params["fx"] = fx
            self.cam_params["fy"] = fy
            # principal point (optical center)
            self.cam_params["cx"] = cx
            self.cam_params["cy"] = cy

    def read_frames(self):
        # Get frames list
        frames = []
        seqs = []
        for sequence in self.sequences:
            frames_dir = os.path.join(self.data_path, sequence, "image_{}".format(self.camera_id), "*.jpg")
            frames_seq = sorted(glob.glob(frames_dir))
            frames = frames + frames_seq
            seqs = seqs + [sequence] * len(frames_seq)
        return frames, seqs

    def read_gt(self):
        # Read ground truth
        if self.read_gt:
            gt = []
            for sequence in self.sequences:
                with open(os.path.join(self.gt_path, sequence + ".txt")) as f:
                    lines = f.readlines()

                # convert poses to float
                for line_idx, line in enumerate(lines):
                    line = line.strip().split()
                    line = [float(x) for x in line]
                    gt.append(line)

        else:  # test data (sequences 11-21)
            gt = None

        return gt

    def create_windowed_dataframe(self, df):
        window_size = self.window_size
        overlap = self.overlap
        windowed_df = pd.DataFrame()
        w_idx = 0

        for sequence in df["sequence"].unique():
            seq_df = df.loc[df["sequence"] == sequence, :].reset_index(drop=True)
            row_idx = 0
            while row_idx + window_size <= len(seq_df):
                rows = seq_df.iloc[row_idx:(row_idx + window_size)].copy()
                rows["w_idx"] = len(rows) * [w_idx]  # add window index column
                row_idx = row_idx + window_size - overlap
                w_idx = w_idx + 1
                windowed_df = pd.concat([windowed_df, rows], ignore_index=True)
        windowed_df.reset_index(drop=True)
        return windowed_df


if __name__ == "__main__":

    # Create dataloader
    preprocess = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183]), ])

    data = KITTI(transform=preprocess, sequences=["04"], window_size=3, overlap=2)
    test_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    timings = np.zeros((len(test_loader), 1))
    for i in range(len(test_loader)):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        imgs, gt = data[i]
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time

    mean_syn = np.sum(timings) / len(test_loader)
    std_syn = np.std(timings)
    print("Mean pre-proc time: ", mean_syn)
    print("Std time: ", std_syn)

# idx = 500
# imgs, gt = data[idx]
# print("imgs.shape: ", imgs.shape)
# print("gt.shape: ", gt.shape)

# img = np.moveaxis(imgs[0, :, :, :], 0, -1)

# # post processing
# channelwise_mean = [0.34721234, 0.36705238, 0.36066107]
# channelwise_std = [0.30737526, 0.31515116, 0.32020183]
# img[:, :, 0] = img[:, :, 0] * channelwise_std[0] + channelwise_mean[0]
# img[:, :, 1] = img[:, :, 1] * channelwise_std[1] + channelwise_mean[1]
# img[:, :, 2] = img[:, :, 2] * channelwise_std[2] + channelwise_mean[2]

# plt.figure()
# plt.imshow((img * 255).astype(int));