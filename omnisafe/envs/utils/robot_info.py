import torch
import numpy as np

ROBOT_BASE_HEIGHT = 0.01 # TODO:
TABLE_HEIGHT = 0.6

HOME_POSE = torch.tensor([np.pi*x/180 for x in [0,-100,120,-15,90,90]] + [0] * 16) 
OLD_HOME_POSE = torch.tensor([np.pi*x/180 for x in [0,-60,75,75,90,0]] + [0] * 4 + [0.5] + [0] * 11) # used for ik
OBJ_INIT_CENTER = torch.tensor([0.6, 0.05])

VISIBLE_RIGID_BODY = ['wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'hand_base_link', 'mcp_joint', 'pip', 'dip', 'fingertip', 'pip_4', 'thumb_pip', 'thumb_dip', 'thumb_fingertip', 'mcp_joint_2', 'pip_2', 'dip_2', 'fingertip_2', 'mcp_joint_3', 'pip_3', 'dip_3', 'fingertip_3']
ROBOT_LINKS = ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link'] + VISIBLE_RIGID_BODY

SAPIEN_DOF_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'j12', 'j13', 'j14', 'j15', 'j1', 'j0', 'j2', 'j3', 'j9', 'j8', 'j10', 'j11', 'j5', 'j4', 'j6', 'j7']
MOVEIT_DOF_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'j1', 'j0', 'j2', 'j3', 'j12', 'j13', 'j14', 'j15', 'j5', 'j4', 'j6', 'j7', 'j9', 'j8', 'j10', 'j11']
MOVEIT_TO_SAPIEN_DOF_MAP = [MOVEIT_DOF_NAMES.index(name) for name in SAPIEN_DOF_NAMES]

RIGID_BODY_PC_FILES = dict(
    wrist_1_link='data/ur_description/meshes/ur5/visual/wrist1.npy',
    wrist_2_link='data/ur_description/meshes/ur5/visual/wrist2.npy',
    wrist_3_link='data/ur_description/meshes/ur5/visual/wrist3.npy',
    hand_base_link='data/ur_description/meshes/leap/palm_lower.npy',
    mcp_joint='data/ur_description/meshes/leap/mcp_joint.npy',
    pip='data/ur_description/meshes/leap/pip.npy',
    dip='data/ur_description/meshes/leap/dip.npy',
    fingertip='data/ur_description/meshes/leap/fingertip.npy',
    pip_4='data/ur_description/meshes/leap/pip.npy',
    thumb_pip='data/ur_description/meshes/leap/thumb_pip.npy',
    thumb_dip='data/ur_description/meshes/leap/thumb_dip.npy',
    thumb_fingertip='data/ur_description/meshes/leap/thumb_fingertip.npy',
    mcp_joint_2='data/ur_description/meshes/leap/mcp_joint.npy',
    pip_2='data/ur_description/meshes/leap/pip.npy',
    dip_2='data/ur_description/meshes/leap/dip.npy',
    fingertip_2='data/ur_description/meshes/leap/fingertip.npy',
    mcp_joint_3='data/ur_description/meshes/leap/mcp_joint.npy',
    pip_3='data/ur_description/meshes/leap/pip.npy',
    dip_3='data/ur_description/meshes/leap/dip.npy',
    fingertip_3='data/ur_description/meshes/leap/fingertip.npy',
)
RIGID_BODY_ORIGIN = dict(
    wrist_1_link=[0,0,0],
    wrist_2_link=[0,0,0],
    wrist_3_link=[0,0,0],
    hand_base_link=[-0.020095249652862544332, 0.025757756134899473244, -0.034722403578460216134],
    mcp_joint=[0.0084069022611744960438, 0.0077662438597169954416, 0.014657354985032912051],
    mcp_joint_2=[0.0084069022611744960438, 0.0077662438597169954416, 0.014657354985032912051],
    mcp_joint_3=[0.0084069022611744960438, 0.0077662438597169954416, 0.014657354985032912051],
    pip=[0.0096433630922713280131, 0.00029999999999998951117, 0.00078403401041737645627],
    pip_2=[0.0096433630922713280131, 0.00029999999999998951117, 0.00078403401041737645627],
    pip_3=[0.0096433630922713280131, 0.00029999999999998951117, 0.00078403401041737645627],
    dip=[0.021133352895225002849, -0.0084321191467048792201, 0.0097850881620952408213],
    dip_2=[0.021133352895225002849, -0.0084321191467048792201, 0.0097850881620952408213],
    dip_3=[0.021133352895225002849, -0.0084321191467048792201, 0.0097850881620952408213],
    fingertip=[0.013286424108533503169, -0.0061142383865419869249, 0.014499999999999497666],
    fingertip_2=[0.013286424108533503169, -0.0061142383865419869249, 0.014499999999999497666],
    fingertip_3=[0.013286424108533503169, -0.0061142383865419869249, 0.014499999999999497666],
    pip_4=[-0.0053566369077286714317, 0.00029999999999999991951, 0.00078403401041737819099],
    thumb_pip=[0.011961920770611186859, -5.3082538364890297089e-16, -0.015852648956664199681],
    thumb_dip=[0.043968715707239175439, 0.057952952973709198625, -0.0086286764493694757122],
    thumb_fingertip=[0.062559538462667388381, 0.078459682911396988469, 0.048992911807332215068]
)

RIGID_BODY_RPY = dict(
    wrist_1_link=[0,0,0],
    wrist_2_link=[0,0,0],
    wrist_3_link=[0,0,0],
    hand_base_link=[0,0,0],
    mcp_joint=[1.6375789613220999807e-15, -1.0210473302491019535e-30, 1.7177968783327987474e-31],
    mcp_joint_2=[1.6375789613220999807e-15, -1.0210473302491019535e-30, 1.7177968783327987474e-31],
    mcp_joint_3=[1.6375789613220999807e-15, -1.0210473302491019535e-30, 1.7177968783327987474e-31],
    pip=[-1.570796326794896558, -1.570796326794896336, 0],
    pip_2=[-1.570796326794896558, -1.570796326794896336, 0],
    pip_3=[-1.570796326794896558, -1.570796326794896336, 0],
    dip=[-3.141592653589793116, 4.5075111242164408299e-32, 4.4395481053923607589e-32],
    dip_2=[-3.141592653589793116, 4.5075111242164408299e-32, 4.4395481053923607589e-32],
    dip_3=[-3.141592653589793116, 4.5075111242164408299e-32, 4.4395481053923607589e-32],
    fingertip=[3.141592653589793116, 1.1993117970061734707e-33, 4.4395481053923607589e-32],
    fingertip_2=[3.141592653589793116, 1.1993117970061734707e-33, 4.4395481053923607589e-32],
    fingertip_3=[3.141592653589793116, 1.1993117970061734707e-33, 4.4395481053923607589e-32],
    pip_4=[-1.570796326794896558, -1.570796326794896336, 0],
    thumb_pip=[1.570796326794896558, 1.6050198443300152637e-46, -3.9204996250525192755e-59],
    thumb_dip=[1.9428902930940098942e-16, 3.2751579226442200773e-15, 1.1123758529657360012e-46],
    thumb_fingertip=[4.3790577010156367543e-47, -3.3306690738754701143e-16, 1.2042408677791935383e-46]
)

RIGID_BODY_BIAS = dict(
    hand_base_link=[-0.04859524965286255, -0.03710692886510053, -0.011722403528460216],
    fingertip=[-0.000799055891466496, -0.03481634338654199, 0.014499999999999501],
    fingertip_2=[-0.000799055891466496, -0.03481634338654199, 0.014499999999999501],
    fingertip_3=[-0.000799055891466496, -0.03481634338654199, 0.014499999999999501],
    thumb_fingertip=[-0.0007990565373326153, -0.046100002088603015, -0.01430000319266779],
)

RIGID_BODY_SIZE = dict(
    fingertip=[0.02059812, 0.02936731, 0.024],
    fingertip_2=[0.02059812, 0.02936731, 0.024],
    fingertip_3=[0.02059812, 0.02936731, 0.024],
    thumb_fingertip=[0.020598110000000003, 0.03200000999999998, 0.024000010000000002],
)

FINGERTIP_BIAS = dict(
    fingertip=[-0.000799055891466496, -0.03481634338654199, 0.014499999999999501],
    fingertip_2=[-0.000799055891466496, -0.03481634338654199, 0.014499999999999501],
    fingertip_3=[-0.000799055891466496, -0.03481634338654199, 0.014499999999999501],
    thumb_fingertip=[-0.0007990565373326153, -0.046100002088603015, -0.01430000319266779],
)

FINGERTIP_NORMAL = dict(
    fingertip=[-1, 0, 0],
    fingertip_2=[-1, 0, 0],
    fingertip_3=[-1, 0, 0],
    thumb_fingertip=[-1, 0, 0],
)

UR5_PARAMS = torch.tensor([
            [0, 0.089159, np.pi/2],
            [-0.425, 0, 0],
            [-0.39225, 0, 0],
            [0, 0.10915, np.pi/2],
            [0, 0.09465, -np.pi/2],
            [0, 0.18192030965286254, 0]
            ])

TACTILE_SENSOR_INFO = dict(
    width=0.0064,
    height=0.001,
    effective=0.005,
)

DISABLE_COLLISION = {
    'mcp_joint': (1, 1),
    'pip': (1, 1),
    'dip': (1, 1),
    'mcp_joint_2': (1, 2),
    'pip_2': (1, 2),
    'dip_2': (1, 2),
    'mcp_joint_3': (1, 3),
    'pip_3': (1, 3),
    'dip_3': (1, 3),
    'wrist_1_link': (1, 4),
    'wrist_3_link': (1, 4),
    'thumb_dip': (1, 5),
    'hand_base_link': (1, 5),
    'thumb_pip': (1, 5),
}