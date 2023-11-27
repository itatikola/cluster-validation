'''
A class used to contain metadata about a single assembly (frame of keypoints of one fish)
and methods to generate statistics about that assembly

'''

import pandas as pd
import numpy as np


class Assembly:

    PARTS = ['nose', 'leftEye', 'rightEye', 'stripe1', 'stripe2', 'stripe3', 'stripe4', 'tailBase', 'tailTip']

    # initializer
    # hdf_path (str) : original h5 file that the assembly is from
    # assembly (DataFrame) : original row of key points
    # frame (?): frame it was from
    def __init__(self, hdf_path, assembly):
        self.original_hdf = hdf_path
        self.assembly = assembly
        self.frame = assembly.name
        self.individual = assembly.index[0][1]
        self.distance_matrix = None
        self.proportions = None

    def prop_dict(self):
        return self.proportions.to_dict()

    def populate_matrix(self, chunk=3):
        self.distance_matrix = pd.DataFrame(index=self.PARTS, columns=self.PARTS)
        # 1. fill with euclidean distances - only those connected by skeleton
        for i in range(0, len(self.assembly) - chunk, chunk):
            part_one = self.assembly.index[i][2]
            part_two = self.assembly.index[i + chunk][2]
            self.distance_matrix[part_one][part_two] = self.euclidean_distance(part_one, part_two)
            self.distance_matrix[part_two][part_one] = self.distance_matrix[part_one][part_two]
        self.distance_matrix['nose']['rightEye'] = self.euclidean_distance('nose', 'rightEye')
        self.distance_matrix['leftEye']['stripe1'] = self.euclidean_distance('leftEye', 'stripe1')
        self.distance_matrix['nose']['stripe1'] = self.euclidean_distance('nose', 'stripe1')
        # 2. fill cumulative spine distances
        for i in range(0, 9):
            part_one = self.PARTS[i]
            self.distance_matrix[part_one][part_one] = 0
            if part_one != 'leftEye' and part_one != 'rightEye':
                self.fill_cumulative_distance(part_one, i)

    def generate_proportions(self):
        if self.distance_matrix is None:
            raise ValueError("Distance matrix has not been populated.")
        indexes = []
        proportions = []
        # 1. adjacent proportions
        spine_length = self.distance_matrix['nose']['tailTip']
        for i in range(1, 8):
            part_one = self.PARTS[i - 1]
            joint = self.PARTS[i]
            part_two = self.PARTS[i + 1]
            indexes.append(part_one + "+" + joint + "_to_" + joint + "+" + part_two)
            proportions.append(self.distance_matrix[part_one][joint] / self.distance_matrix[joint][part_two])
            indexes.append(part_one + "+" + joint + "_to_spineLength")
            proportions.append(self.distance_matrix[part_one][joint] / spine_length)
        # 2. manual skeleton
        indexes.append("nose+leftEye_to_nose+rightEye")
        proportions.append(self.distance_matrix['nose']['leftEye'] / self.distance_matrix['nose']['rightEye'])
        indexes.append("leftEye+stripe1_to_rightEye+stripe1")
        proportions.append(self.distance_matrix['leftEye']['stripe1'] / self.distance_matrix['rightEye']['stripe1'])
        indexes.append("nose+stripe1_to_leftEye+rightEye")
        proportions.append(self.distance_matrix['nose']['stripe1'] / self.distance_matrix['leftEye']['rightEye'])
        indexes.append("tailBase+tailTip_to_spineLength")
        proportions.append(self.distance_matrix['tailBase']['tailTip'] / spine_length)
        self.proportions = pd.Series(proportions)
        self.proportions.index = indexes

    def euclidean_distance(self, part_one, part_two):
        gen_index = self.assembly.index[0]
        vec1 = np.array([self.assembly.__getitem__((gen_index[0], gen_index[1], part_one, 'x')),
                         self.assembly.__getitem__((gen_index[0], gen_index[1], part_one, 'y'))])
        vec2 = np.array([self.assembly.__getitem__((gen_index[0], gen_index[1], part_two, 'x')),
                         self.assembly.__getitem__((gen_index[0], gen_index[1], part_two, 'y'))])
        return np.linalg.norm(vec2 - vec1)

    def fill_cumulative_distance(self, part_one, i):
        to_start = 2
        if part_one == 'nose':
            to_start = 4
        for j in range(i + to_start, 9):
            part_two = self.PARTS[j]
            if part_two != 'leftEye' and part_two != 'rightEye':
                self.distance_matrix[part_one][part_two] = self.distance_matrix[part_one][self.PARTS[j - 1]] \
                                                           + self.distance_matrix[self.PARTS[j - 1]][part_two]
                self.distance_matrix[part_two][part_one] = self.distance_matrix[part_one][part_two]


