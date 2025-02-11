import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List,Any
import functools
from datetime import datetime
import dill



import matplotlib.pyplot as plt
import jax
from jax import numpy as jp
import numpy as np
from matplotlib import pyplot as plt
from orbax import checkpoint as ocp


import os
import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model




class H1(PipelineEnv):
    def __init__(
      self,
      obs_noise: float = 0.05,
      action_scale: float = 0.3,
      disturbance_vel: float =0.05,
      contact_limit: float = 0.021,# distance smalled than which we consider there is contact between the feet and the ground
      done_limit: float = 0.6,# when the robot is falling over, might need to change depending 
                              #on the movemnt(walking, picking something up)
      
      timestep: float = 0.025,# todo :not sure if it is used or needed
      scene_file: str = 'unitree_h1/scene.xml',
      **kwargs,):
        
        path = os.path.join(os.getcwd(),scene_file)
        sys = mjcf.load(path)
        self._dt = 0.02  # this environment is 50 fps this is how often the agent interacts with the simulation
        sys = sys.tree_replace({'opt.timestep': 0.004}) # sets the simulation timestep,check https://mujoco.readthedocs.io/en/stable/XMLreference.html#option

        
        n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep)) 
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        # store observation noise coefficient
        self.obs_noise = obs_noise
        # store disturbance velocity
        self.disturb_vel = disturbance_vel
        # store contact limit
        self.contact_limit = contact_limit
        # store done limit
        self.done_limit = done_limit
        # store timestep
        self.timestep = timestep
        # store action scale
        self.action_scale = action_scale

        # store site position for contact detection
        feet_site = ["left_foot_back","left_foot_front","right_foot_back","right_foot_front"]
        feet_site_id = [mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_site] #todo : what does value does here, seems the same results without it
        #feet_site_id2 = [sys.mj_model.site(f).id for f in feet_site]
        #print(feet_site_id2)
        #print(sys.mj_model.site("left_foot_back").pos)
        self.feet_site_id = jp.array(feet_site_id)
        #print(sys.mj_model.joint("free_joint"))



        #todo : difference between actuator range and joint range
        #todo : many ways to access the same thing, as  long as this thing is the model or its data
        self.actuator_range=sys.actuator.ctrl_range
        #self.actuator_range = sys.actuator_ctrlrange
        #print(sys.qpos0)
        #print(sys.mj_model.actuator_ctrlrange) 
        #print(sys.mj_model.qpos0)

        
        self.inital_qpos = jp.array(sys.mj_model.keyframe('home').qpos) #todo: define the home keyframe
        #print(len(self.inital_qpos))

        ###############
        ###joint position and control 
        #free joint 7:3 pos and 4 quaternion orinetaion  

        #actuator_names=["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle","right_hip_yaw","right_hip_roll",
        #        "right_hip_pitch","right_knee","right_ankle","torso","left_shoulder_pitch","left_shoulder_roll",
        #        "left_shoulder_yaw","left_elbow","right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow"
        #        
        #        ]
        #motor_id=[sys.mj_model.joint(f).id for f in actuator_names]
        #print(motor_id)
        #todo : the joints where we have an actuator
        self.default_jnt_angle = jp.concatenate([
            sys.mj_model.keyframe('home').qpos[0:3],
            sys.mj_model.keyframe('home').qpos[7:]
        ]) #todo: might bug need to check
        self.motor_angle = jp.concatenate([ 
            sys.mj_model.keyframe('home').qpos[7].reshape(-1),
            sys.mj_model.keyframe('home').qpos[8].reshape(-1),
            sys.mj_model.keyframe('home').qpos[9].reshape(-1),
            sys.mj_model.keyframe('home').qpos[10].reshape(-1),
            sys.mj_model.keyframe('home').qpos[11].reshape(-1),
            sys.mj_model.keyframe('home').qpos[12].reshape(-1),
            sys.mj_model.keyframe('home').qpos[13].reshape(-1),
            sys.mj_model.keyframe('home').qpos[14].reshape(-1),
            sys.mj_model.keyframe('home').qpos[15].reshape(-1),
            sys.mj_model.keyframe('home').qpos[16].reshape(-1),
            sys.mj_model.keyframe('home').qpos[17].reshape(-1),
            sys.mj_model.keyframe('home').qpos[18].reshape(-1),
            sys.mj_model.keyframe('home').qpos[19].reshape(-1)
        ]) 
        #print(self.motor_angle)
        #print(len(sys.mj_model.jnt_range[]))
        #todo: add joint limitis if want to implement a negative reward for getting closr to joint limits
        self.standing = sys.mj_model.keyframe('home').qpos[7:] #todo: this should work because we don't have ball joints or we would
                                                                    #need to save and use ctrl since the dims of this and action would be diifernt


        # store the degree of freedom
        self.nv = sys.nv
        # store the number of actuators
        self.nu = sys.nu
        self.pelvis_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis') 

        self.left_foot_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_link') 
        self.right_foot_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_link') # todo: is it really needed for the obs
        self.foot_id = jp.array([self.left_foot_id,self.right_foot_id]) 

        self.jnt_size = len(self.default_jnt_angle)



    
    def step(self, state:State, action:jax.Array):

        rng, ctl_rng, disturb_rng = jax.random.split(state.info['Random generator'],3)
        
        
        # Action scale and step

        # Apply the action to the state and step it forward. The action should be scaled to its
        # actuators range

        # scale the action to actuators range
        action = self._action_scale(action)

        # action the motor to its default pose
        motor_action = self.standing + action * self.action_scale

        # update the state
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_action)
        #Pipelien state and State are differennt, in the github State defined in brax/envs/base incldes pipelinesate which is defined in brax/base
        #pipelinesate contains info about the simulation angles,velocirites,x and dx
        #state contains info about the "state/observation"



        # State information extraction

        # Extract the information:
        # body_pos: position or orientation of the robot
        # body_vel: derivative of x, velocity or angular velocity of the robot
        # joint_angle: joint angle of the robot (only important joint)
        # limited_angle: joint angle of the robot with limit
        # joint_velocity: joint velocity of the robot
        body_pos = pipeline_state.x
        body_vel = pipeline_state.xd

        limited_angle = jp.concatenate([pipeline_state.q[7:]])
        #todo: chec if the joints velocities are needed
        # joint_velocity = jp.concatenate([ 
        #     pipeline_state.qd[0:3],
        #     pipeline_state.qd[6:9],
        #     pipeline_state.qd[12:15],
        #     pipeline_state.qd[18].reshape(-1),
        #     pipeline_state.qd[19:22],
        #     pipeline_state.qd[25:28],
        #     pipeline_state.qd[31].reshape(-1)
        # ])
        #print(limited_angle)
        # Observation

        # Get the observation of the environment
        obs = self._get_obs(pipeline_state, state.info)

        
        ###########################
        #calculate AIR time
        ###########################

        foot_pos = pipeline_state.site_xpos[self.feet_site_id]
        foot_pos = foot_pos[:,2]

        left_foot_pos = foot_pos[0:2]
        right_foot_pos = foot_pos[2:4]

        left_contact = jp.all(left_foot_pos < 0.021)
        right_contact = jp.all(right_foot_pos < 0.021)

        # check if the feet are in contact with the ground
        contact = jp.array([left_contact, right_contact])

        # combine the current contact status with the last contact stauts
        # to have a smoother transition
        contact_filt = contact | state.info['Last Contact']

        # identify if it is the first contact after in the air
        first_contact = (state.info['Feet air time']>0) * contact_filt

        # update the feet air time
        state.info['Feet air time'] += self.timestep#todo: simestep or dt







        ###########################
        #Termination of epoisode#
        ###########################
        done = pipeline_state.x.pos[self.pelvis_id-1,2] < 0.6 #todo: define the done condition and check if there are situation for terminatin joint limits


        ###########################
        #todo: calculate the orienaion for some tpes of rewards
        orientation = self._calculate_forward_orientation(state.info['Control commands'])


        ###########################
        #Reward
        ###########################
        #Calculate the reward whcih will appended to the info of the state
        joint_angle = jp.concatenate([
            pipeline_state.q[0:3],
            pipeline_state.q[7:] #todo: might bug need to check
        ])

        """rewards = {
            'tracking linear velocity reward':(
                2*self._linear_velocity_tracking(state.info['Control commands'],body_vel)),
        
            # 'tracking angular velocity reward':(
            #     1.5*self._angular_velocity_tracking(state.info['Control commands'],body_vel)),

            'z axis velocity penalty':(
                -1*self._linear_velocity_penalty(body_vel)),

            'x-y plane angular velocity penalty':(
                -0.5*self._angular_velocity_penalty(body_vel)),

            'joint torque penalty':(
                -0.00005*self._joint_torque_penalty(pipeline_state.qfrc_actuator)),

            'action rate penalty':(
                -0.1*self._action_rate_penalty(action,state.info['Last action'])),

            'offset when standing still penalty':(
                -1*self._motion_standing_still_penalty(state.info['Control commands'],joint_angle)),

            # 'alive reward':(
            #     3*self._alive_reward(done,state.info['Step'])),

            'feet air time reward':(
                8*self._feet_air_time_reward(state.info['Feet air time'],first_contact,state.info['Control commands'])),

            'Single feet on ground reward':(
                0.3*self._single_leg_on_ground_reward(contact,state.info['Control commands'])),

            'early termination penalty':(
                -1*self._early_termination_penalty(done,state.info['Step total'])),
            
            # 'Both leg in the air penalty':(
            #     -2*self._both_foot_air_penalty(contact)),

            #'joint limit penalty':(
            #    -2*self._joint_limit_penalty(limited_angle)),

            'orientation penalty':(
                -1*self._orientation_penalty(body_pos)),

            #'feet position penalty':(
            #    -2*self._feet_position_penalty(body_pos)),

            #'z position penalty':(
            #    -2*self._z_position_penalty(body_pos)),

            'orientation walking penalty':(
                -3*self._orientation_walking_penalty(orientation,body_pos))
             
        }"""
        rewards={'rew':1}

        # calculate the total reward
        reward = sum(rewards.values())



        # State information update
    
        # update state.info to track the training progress and gather useful infomation

        # update the state information
        #state.info['State'] = pipeline_state
        state.info['Random generator'] = rng
        state.info['Last action'] = action
        #state.info['Last pelvis position'] = body_pos.pos[self.pelvis_id-1]
        #state.info['Last pelvis velocity'] = body_vel.vel[self.pelvis_id-1]
        #state.info['Last joint angle'] = joint_angle
        state.info['Last Contact'] = contact
        state.info['Feet air time'] *= ~contact_filt
        state.info['Reward'] = reward
        # state.info['Reward dict'] = rewards
        state.info['Step'] += 1
        state.info['Step total'] += 1 #todo: inn reset it is set to zero, maybe could remove no use or it measure the len of episode the other when to change command
        state.info['Total distance'] = math.normalize(body_pos.pos[self.pelvis_id-1][:2])[1]

        # update the control commands when more than 500 timestep has achieved
        state.info['Control commands'] = jp.where(
        # condition: step>500
            state.info['Step'] > 500,
        # if true
            self.control_commands(ctl_rng), #todo implemt control command
        # if false
            state.info['Control commands']
        )
    
        # reset the step counter when the episode is terminated or reached 500 steps
        state.info['Step'] = jp.where(
        # condition: done or step>500
            done | (state.info['Step'] > 500),
        # if true
            0,
        # if false
            state.info['Step']
        )

        # log total displacement as a proxy metric
        state.metrics['Total distance'] = state.info['Total distance']
        # update the reward into proxy metric
        state.metrics['reward'] = reward
        # convert termination flag
        done = jp.float32(done)

    
        state = state.replace(
            pipeline_state = pipeline_state,
            obs = obs,
            reward = reward,
            done = done,
        )

        return state
    
    def reset(self, rng:jax.Array):
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self.inital_qpos,jp.zeros(self.nv))
        state_info = {

            #'State': pipeline_state,
            'Random generator': rng,
            'Control commands': self.control_commands(key),
            'Last action': jp.zeros(self.nu),
            #'Last pelvis position': jp.zeros(3),
            #'Last pelvis velocity': jp.zeros(3),
            #'Last joint angle': jp.zeros(self.jnt_size),
            'Last Contact': jp.zeros(2,dtype=bool),
            'Feet air time': jp.zeros(2),
            'Reward': 0.0,
            'Step': 0,
            'Step total': 0,
            # 'Reward dict': {}
            'Total distance': 0.0
            
        }



        reward, done = jp.zeros(2)

        # initialize additional metrics
        metrics = {'Total distance': 0.0,
                   'reward':0.0}
        

        """ joint_velocity = jp.concatenate([
            pipeline_state.qd[0:3],
            pipeline_state.qd[6:9],
            pipeline_state.qd[12:15],
            pipeline_state.qd[18].reshape(-1),
            pipeline_state.qd[19:22],
            pipeline_state.qd[25:28],
            pipeline_state.qd[31].reshape(-1)
        ]) """

        obs = self._get_obs(pipeline_state, state_info)

        # construct the state
        state= State(
            pipeline_state = pipeline_state,
            obs = obs,
            reward = reward,
            done = done,
            metrics = metrics,
            info = state_info
        )

        return state
    



    ###########################
    #          Observation
    ###########################

    def _get_obs(
            self,
            pipeline_state: State,
            state_info: dict[str,Any]
    ) -> jax.Array:
        """
        Get the observation of the environment.
        
        input: 
            pipeline_state: the state of the robot
            state_info: the information of the state
            joint_angle: the joint angle of the robot
            joint_velocity: the joint velocity of the robot

        output: 
            obs: the observation of the environment
        """
        # calculate the inverse of quaternion of pelvis
        inv_pelvis_rot = math.quat_inv(pipeline_state.x.rot[self.pelvis_id-1])
        #print("joint_angle   ",joint_angle)
        #print("df joint      ",self.default_jnt_angle)
        # create observation vector
        joint_angle = jp.concatenate([
            pipeline_state.q[0:3],
            pipeline_state.q[7:] #todo: might bug need to check
        ])

        obs = jp.concatenate([
            # pelvis linear velocity
            pipeline_state.xd.vel[self.pelvis_id-1]*2.0,

            # pelvis angular velocity
            pipeline_state.xd.ang[self.pelvis_id-1]*0.25,

            # direction of gravity relative to pelvis orientation
            math.rotate(jp.array([0,0,-1]),inv_pelvis_rot),

            # control command
            (state_info['Control commands'] * jp.array([2.0,2.0])).flatten(), #todo: why flatten

            # joint angle difference
            joint_angle - self.default_jnt_angle,

            # joint velocity
            #joint_velocity * 0.05,

            # action
            state_info['Last action'],

            # feet position
            pipeline_state.xpos[self.foot_id].flatten()
            ])
        # clip the observation to prevent extreme values
        # add noise to encourage robustness
        obs = obs + self.obs_noise * jax.random.uniform(
            state_info['Random generator'], shape=obs.shape ,minval=-1, maxval=1)

        return obs
    



    ###########################
    #Control commands   todo: add ang velocity, so robot can rotate
    ###########################
    def control_commands(
            self,
            rng:jax.Array,
    ) -> jax.Array:
        """
        Set the random velocity reference for the agent to track.
        
        input: 
            rng: random key generator from Jax

        output: 
            col_command: the random velocity reference for the agent to track
        
        """
        # set the random keys
        key1, key2, key3 = jax.random.split(rng,3)

        # set the control commands limitation
        velocity_x_limit = [0.0, 1.5]
        velocity_y_limit = [-0.5, 0.5]
        # angular_velocity_limit = [-0.5, 0.5]

        # set the random values for velocity
        velocity_x_command=jax.random.uniform(key1,shape=(1,),minval=velocity_x_limit[0],maxval=velocity_x_limit[1])
        velocity_y_command=jax.random.uniform(key2,shape=(1,),minval=velocity_y_limit[0],maxval=velocity_y_limit[1])    
        # angular_velocity_command=jax.random.uniform(key3,shape=(1,),minval=angular_velocity_limit[0],maxval=angular_velocity_limit[1])

        # combine the values
        col_command = jp.array([velocity_x_command[0],velocity_y_command[0]])
        return col_command
    
    #                                   action scale
    def _action_scale(
            self, 
            action: jax.Array
    ) -> jax.Array:
        """
        Scale the action to the actuators range. The formula fron interval [-1,1] to [a,b] is:

        y = (x+1)*(b-a)/2 + a

        input: 
            action: the action to scale
        
        output:
            scaled_action: the scaled action
        """

        # define new interval range
        a = self.actuator_range[:,0]
        b = self.actuator_range[:,1]

        # scale the action
        scaled_action = (action+1)*(b-a)/2 + a
        
        return scaled_action
    #                                   forward orientation checking
    def _calculate_forward_orientation(
            self,
            command:jax.Array,
            tolerance = 1e-7
    ) -> jax.Array:
        """
        Calculate the forward orientation of the robot

        input: 
            command: the control command of the robot

        output:
            orientation: the forward orientation of the robot
        """
        # get velocity from command
        velocity_x, velocity_y = command

        # calculate the angle in radians
        theta = jp.arctan2(velocity_y, velocity_x)  

        # calculate the cosine of the angle
        cos_theta = jp.cos(theta)  
        # calculate the sine of the angle
        sin_theta = jp.sin(theta)  

         # set small values to zero
        cos_theta = jp.where(jp.abs(cos_theta) < tolerance, 0, cos_theta)
        sin_theta = jp.where(jp.abs(sin_theta) < tolerance, 0, sin_theta)

        # create orientation vector
        orientation = jp.array([cos_theta,sin_theta])

        return orientation
    #                                 
    # 
    # 
    #  Reward functions


    def _linear_velocity_tracking(self, #todo: might need to use angular from command too, if not then might need to remove this reward
                                  command:jax.Array,
                                  body_vel:Motion) -> jax.Array:
        """
        reward term for tracking the reference linear velocity

        formula:
            q_linear = exp(-error^2/0.25)
        """

        # calculate the local velocity of robots from global velocity
        # local_vel = math.rotate(body_vel.vel[self.pelvis_id-1],math.quat_inv(body_pos.rot[self.pelvis_id-1]))

        # calculate the error
        linear_vel_err = jp.sum(jp.square(command[:2] - body_vel.vel[self.pelvis_id-1][:2]))
        #todo: chec this equation might need to uncomment local_vel
        # calculate the reward term
        reward = jp.exp(-linear_vel_err/0.25)

        return reward
        
    def _linear_velocity_penalty(self,
                                 body_vel:Motion) -> jax.Array:
        """
        penalty term for z axis velocity of pelvis

        provide steady walking and pace

        formula:
            q_z = -velocity_z^2
        """
        # calculate reward
        reward = jp.square(body_vel.vel[self.pelvis_id-1,2])

        return reward
        
    
    
    
    def _angular_velocity_penalty(self,
                                  body_vel:Motion) -> jax.Array:
        """
        penalty term for x-y plane angular velocity

        increase stability and reduce spinning

        formula:
            q_ang = -angular_velocity^2
        """
        # calculate reward
        reward = jp.sum(jp.square(body_vel.ang[self.pelvis_id-1,:2]))

        return reward

        
    def _joint_torque_penalty(self,
                              joint_torque:jax.Array) -> jax.Array:
        """
        penalty term for L2 norm of the total torques

        L2 norm: square root of the sum of the squares of the elements
        it provides a single scalar reprsents the magnitude of the torque vector
        in this case it means the strength or energy of the torques applied by all actuators combined
        this can encourage the robot to use less overall torque to achieve its objectives
        therefore discourage from applying large torques across multiple actuators

        """
        # calculate L2 norm of torque
        L2_norm = jp.sqrt(jp.sum(jp.square(joint_torque)))
        
        # calculate reward
        reward = L2_norm

        return reward
    
    def _motion_standing_still_penalty(self,
                               command:jax.Array,
                               joint_angle:jax.Array) -> jax.Array:
        """
        penalty term for offset of robot when no command

        encourage robot to stand still when no command

        formula:
            q_still = -offset^2
        """
        # calculate the error
        # the second term represent when small or no command are given
        reward = jp.mean((joint_angle - self.default_jnt_angle)**2) * (math.normalize(command[:2])[1] < 0.1)

        return reward



    def _feet_air_time_reward(self,
                              air_time:jax.Array,
                              first_contact:jax.Array,
                              command:jax.Array) -> jax.Array:
        """
        reward term for spending time in the air, encouraging taking steps

        formula:
            q_air = sum(t_air - 0.5)
        """

        # obtain the reward when it making the first contact
        reward = jp.sum((air_time - 0.5) * first_contact)

        # check if there is a command
        reward *= (math.normalize(command[:2])[1] > 0.1)

        return reward
    
    def _single_leg_on_ground_reward(self,
                                     contact:jax.Array,
                                     command:jax.Array) -> jax.Array:
        """
        reward term for encouraging the agent to keep one foot in contact with ground at all time
        """
        # check if single_contact
        singe_contact = jp.sum(contact) == 1

        # update reward
        reward = 1.0*singe_contact*(math.normalize(command[:2])[1] > 0.1)

        return reward
    


    def _early_termination_penalty(self,
                                   done:jax.Array,
                                   step:jax.Array) -> jax.Array:
        """
        penalty term for early termination

        discourage the agent from terminating the episode early
        """
        # calculate reward
        terminal_early = done * (step < 950)
        reward = (950 - step) * terminal_early

        return reward
    

    def _orientation_walking_penalty(self,
                                     orientation:jax.Array,
                                     body_pos:Transform) -> jax.Array:
        """
        penalty term for orientation of the robot
        """
        # define global y-axis
        global_x = jp.array([1.0,0.0,0.0])

        # calculate the local y-axis
        local_x = math.rotate(global_x,body_pos.rot[self.pelvis_id-1])

        # ignore z-axis
        local_x = local_x[:2]

        # calculate the error
        reward = jp.sum(jp.abs(orientation - local_x))

        return reward
    
    def _orientation_penalty(self,
                             body_pos:Transform) -> jax.Array:
        """
        penalty term for orientation of the robot
        """
        # define global up direction
        up = jp.array([0.0,0.0,1.0])

        # calculate the local up direction
        rot_up = math.rotate(up,body_pos.rot[self.pelvis_id-1])

        # calculate the error
        reward = jp.sum(jp.square(rot_up[:2]))

        return reward
    
    def _action_rate_penalty(self,
                             action:jax.Array,
                             last_action:jax.Array) -> jax.Array:
        """
        penalty term for the rate of change of the action

        discourage agressive action and encourage smooth control

        formula:
            q_rate = -(action-last_action)^2
        """
        # calculate reward
        reward = jp.mean((action - last_action)**2)

        return reward



    def _alive_reward(self,
                    done:jax.Array) -> jax.Array:
        """
        reward term for rlive

        encourage the robot to be alive longer
        """
        return 1-done

envs.register_environment('H1', H1)
env_name = 'H1'




env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))

#pre_model_path = os.path.join(os.getcwd(),"Code/policy_walking_final")
#pre_model = model.load_params(pre_model_path)
# previous_params=pre_model
train_fn = functools.partial(
      ppo.train, num_timesteps=1000000,num_evals=10, #increase the number of steps
      reward_scaling=1, episode_length=1000, normalize_observations=True,
      action_repeat=1, unroll_length=20, num_minibatches=32,
      num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
      entropy_cost=1e-2, num_envs=8196, batch_size=256,
      network_factory=make_networks_factory)#todo add previous model

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

def progress(num_steps, metrics):
    print("num_steps    ",num_steps)
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    plt.xlim([0, train_fn.keywords['num_timesteps']])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')
    plt.plot(x_data, y_data)
    plt.show()

make_inference_fn, params, _= train_fn(environment=env,
                                       progress_fn=progress,
                                       eval_env=eval_env)

# save params
model_path = 'Code/policy_walking_final1'
model.save_params(model_path, params)
full_path = 'Code/inference_walking_final1'
# save inference func
with open(full_path, 'wb') as f:
    dill.dump(make_inference_fn, f)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')


        
        