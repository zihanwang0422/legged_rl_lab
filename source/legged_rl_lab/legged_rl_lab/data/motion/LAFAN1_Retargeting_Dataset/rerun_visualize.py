import argparse
import time
import numpy as np
import pinocchio as pin
import rerun as rr
import trimesh

class RerunURDF():
    def __init__(self, robot_type):
        self.name = robot_type
        match robot_type:
            case 'g1':
                self.robot = pin.RobotWrapper.BuildFromURDF('robot_description/g1/g1_29dof_rev_1_0.urdf', 'robot_description/g1', pin.JointModelFreeFlyer())
                self.Tpose = np.array([0,0,0.785,0,0,0,1,
                                       -0.15,0,0,0.3,-0.15,0,
                                       -0.15,0,0,0.3,-0.15,0,
                                       0,0,0,
                                       0, 1.57,0,1.57,0,0,0,
                                       0,-1.57,0,1.57,0,0,0]).astype(np.float32)
            case 'h1_2':
                self.robot = pin.RobotWrapper.BuildFromURDF('robot_description/h1_2/h1_2_wo_hand.urdf', 'robot_description/h1_2', pin.JointModelFreeFlyer())
                assert self.robot.model.nq == 7 + 12+1+14
                self.Tpose = np.array([0,0,1.02,0,0,0,1,
                                       0,-0.15,0,0.3,-0.15,0,
                                       0,-0.15,0,0.3,-0.15,0,
                                       0,
                                       0, 1.57,0,1.57,0,0,0,
                                       0,-1.57,0,1.57,0,0,0]).astype(np.float32)
            case 'h1':
                self.robot = pin.RobotWrapper.BuildFromURDF('robot_description/h1/h1.urdf', 'robot_description/h1', pin.JointModelFreeFlyer())
                assert self.robot.model.nq == 7 + 10+1+8
                self.Tpose = np.array([0,0,1.03,0,0,0,1,
                                       0,0,-0.15,0.3,-0.15,
                                       0,0,-0.15,0.3,-0.15,
                                       0,
                                       0, 1.57,0,1.57,
                                       0,-1.57,0,1.57]).astype(np.float32)
            case _:
                print(robot_type)
                raise ValueError('Invalid robot type')
        
        # print all joints names
        # for i in range(self.robot.model.njoints):
        #     print(self.robot.model.names[i])
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.update()
    
    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh
   
    def load_visual_mesh(self):       
        self.robot.framesForwardKinematics(pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]
            
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))
            
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                   rr.Mesh3D(
                       vertex_positions=mesh.vertices,
                       triangle_indices=mesh.faces,
                       vertex_normals=mesh.vertex_normals,
                       vertex_colors=mesh.visual.vertex_colors,
                       albedo_texture=None,
                       vertex_texcoords=None,
                   ),
                   static=True)
    
    def update(self, configuration = None):
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='dance1_subject2')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    args = parser.parse_args()

    rr.init(
        'Reviz', 
        spawn=True
    )
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    file_name = args.file_name
    robot_type = args.robot_type
    csv_files = robot_type + '/' + file_name + '.csv'
    data = np.genfromtxt(csv_files, delimiter=',')

    rerun_urdf = RerunURDF(robot_type)
    for frame_nr in range(data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        configuration = data[frame_nr, :]
        rerun_urdf.update(configuration)
        time.sleep(0.03)
