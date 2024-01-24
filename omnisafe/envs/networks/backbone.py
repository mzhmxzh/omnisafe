"""
Last modified date: 2023.08.22
Author: Jialiang Zhang
Description: vision backbone
"""

import torch
from networks.pointnet import PointNet
try:
    from networks.sparseconv import MinkowskiFCNN
except:
    pass
from networks.mlp import MLP


class FeatureExtractor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.robot_mlp = MLP(**config.robot_mlp_parameters)

        # build vision backbone
        self.vision_backbone = dict(
            mock=MockBackbone, 
            single=SinglePointNet, 
            seperate=SeperatePointNet, 
            sparseconv=SparseConv, 
        )[config.vision_backbone_type](**config.vision_backbone_parameters)
        if config.freeze_vision_backbone:
            self.vision_backbone.requires_grad_(False)
            print('freezing vision backbone')
        self.observation_dim = config.robot_mlp_parameters['output_dim'] + self.vision_backbone.visual_feature_dim
    
    @staticmethod
    def embed_robot_state(robot_state):
        return robot_state
        # only embed the angles, but also retain the original angles
        # return torch.cat([robot_state[..., :22].cos(), robot_state[..., :22].sin(), robot_state], dim=-1)
    
    def forward(self, robot_state_stacked, visual_observation):
        robot_feature = self.robot_mlp(self.embed_robot_state(robot_state_stacked.reshape(-1, robot_state_stacked.shape[2])[:, :self.config.robot_mlp_parameters.input_dim])).reshape(robot_state_stacked.shape[0], -1)
        visual_feature, keypoints = self.vision_backbone(visual_observation)
        visual_feature = visual_feature.reshape(robot_state_stacked.shape[0], -1)
        if keypoints is not None:
            keypoints = keypoints.reshape(robot_state_stacked.shape[0], -1)
        observation_feature = torch.cat([robot_feature, visual_feature], dim=1)
        return observation_feature, keypoints

        

class MockBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_feature_dim = 0
    
    def forward(self, mock_visual_observation):
        return mock_visual_observation, None


class SinglePointNet(torch.nn.Module):
    def __init__(
        self, 
        scene_pn_parameters=dict(
            point_feature_dim=4,
            local_conv_hidden_layers_dim=[64, 128, 256], 
            global_mlp_hidden_layers_dim=[256, 128], 
            pc_feature_dim=128, 
            activation='leaky_relu'
        ), 
        scene_encoder_path=None, 
    ):
        super().__init__()
        # build network
        self.visual_feature_dim = scene_pn_parameters['pc_feature_dim']
        self.scene_pn = PointNet(**scene_pn_parameters)
        # load checkpoint
        if scene_encoder_path is not None:
            checkpoint = torch.load(scene_encoder_path)
            self.scene_pn.load_state_dict(checkpoint['encoder'])
            print('using pretrained scene pc encoder')
            print(f'model_path: {scene_encoder_path}, iter: {checkpoint["iter"]}')
    
    def forward(self, pc):
        visual_feature = self.scene_pn(pc)
        return visual_feature


class SeperatePointNet(torch.nn.Module):
    def __init__(
        self, 
        robot_pn_parameters=dict(
            point_feature_dim=3, 
            local_conv_hidden_layers_dim=[64, 128, 256], 
            global_mlp_hidden_layers_dim=[256], 
            pc_feature_dim=128, 
        ), 
        object_pn_parameters=dict(
            point_feature_dim=3, 
            local_conv_hidden_layers_dim=[64, 128, 256], 
            global_mlp_hidden_layers_dim=[256], 
            pc_feature_dim=128, 
        ), 
        scene_pn_parameters=dict(
            point_feature_dim=3, 
            local_conv_hidden_layers_dim=[64, 128, 256], 
            global_mlp_hidden_layers_dim=[256, 128], 
            pc_feature_dim=128, 
        ), 
        robot_encoder_path=None, 
        object_encoder_path=None, 
        scene_encoder_path=None, 
    ):
        super().__init__()
        self.visual_feature_dim = robot_pn_parameters['pc_feature_dim'] + object_pn_parameters['pc_feature_dim'] + scene_pn_parameters['pc_feature_dim']
        # build networks
        self.robot_pn = PointNet(**robot_pn_parameters)
        self.object_pn = PointNet(**object_pn_parameters)
        self.scene_pn = PointNet(**scene_pn_parameters)
        # load checkpoints
        if robot_encoder_path is not None:
            checkpoint = torch.load(robot_encoder_path)
            self.robot_pn.load_state_dict(checkpoint['encoder'])
            print('using pretrained robot pc encoder')
            print(f'model_path: {robot_encoder_path}, iter: {checkpoint["iter"]}')
        if object_encoder_path is not None:
            checkpoint = torch.load(object_encoder_path)
            self.object_pn.load_state_dict(checkpoint['encoder'])
            print('using pretrained object pc encoder')
            print(f'model_path: {object_encoder_path}, iter: {checkpoint["iter"]}')
        if scene_encoder_path is not None:
            checkpoint = torch.load(scene_encoder_path)
            self.scene_pn.load_state_dict(checkpoint['encoder'])
            print('using pretrained scene pc encoder')
            print(f'model_path: {scene_encoder_path}, iter: {checkpoint["iter"]}')
    
    def forward(self, pc):
        robot_pc, object_pc = pc
        scene_pc = torch.cat([robot_pc, object_pc], dim=1)
        scene_pc_feature = self.scene_pn(scene_pc)
        object_pc_feature = self.object_pn(object_pc)
        robot_pc_feature = self.robot_pn(robot_pc)
        return torch.cat([scene_pc_feature, object_pc_feature, robot_pc_feature], dim=-1)


class SparseConv(torch.nn.Module):
    def __init__(
        self, 
        minkowski_fcnn_parameters=dict(
            in_channel=4, 
            out_channel=128, 
            embedding_channel=1024, 
            channels=(32, 48, 64, 96, 128), 
            D=3, 
        ), 
        scene_encoder_path=None, 
    ):
        super().__init__()
        self.visual_feature_dim = minkowski_fcnn_parameters['out_channel']
        # build minkowski fcnn
        self.net = MinkowskiFCNN(**minkowski_fcnn_parameters)
        # load checkpoint
        if scene_encoder_path is not None:
            checkpoint = torch.load(scene_encoder_path)
            self.net.load_state_dict(checkpoint['encoder'])
            print('using pretrained scene pc encoder')
            print(f'model_path: {scene_encoder_path}, iter: {checkpoint["iter"]}')
    
    def forward(self, sparse_voxels):
        visual_feature = self.net(sparse_voxels)
        return visual_feature
