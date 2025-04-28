from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Key
from minigrid.minigrid_env import MiniGridEnv
from minigrid.manual_control import ManualControl
import random
from copy import deepcopy
import numpy as np


class PickupOneRoomEnv(MiniGridEnv):
    def __init__(self, max_steps=35, size = 15, agent_view_size=25, **kwargs):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.size = size
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )

        self.world_objects = []
        self.num_objects = 5
        self.agent_type = "teacher"

    @staticmethod
    def _gen_mission():
        return ""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if self.carrying:
            terminated = True
            info["event"] = self.carrying.type + "_" + self.carrying.color
            # self.carrying = False
            # reward = 1
        else:
            info["event"] = ""

        self.max_episode_steps = self.max_steps

        # print(info["event"])

        return obs["image"], reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        self.max_episode_steps = self.max_steps
        return obs["image"], {}

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.agent_type == "teacher":
            self.world_objects = []

        if self.agent_type == "teacher":
            self.reset_pos = deepcopy(self.place_agent(rand_dir=True))
            self.reset_dir = deepcopy(self.agent_dir)
            self.world_objects.append(["agent", self.reset_pos, self.reset_dir])
        else:
            # find the agent in the list of objects
            for i, obj in enumerate(self.world_objects):
                if obj[0] == "agent":
                    self.reset_pos = obj[1]
                    self.reset_dir = obj[2]
                    break
            self.agent_pos = self.reset_pos
            self.agent_dir = self.reset_dir
        
        # place objects
        if self.world_objects is None or self.agent_type == "teacher":
            self.world_objects += self.place_random_objects(self.num_objects)
        else:
            # Place objects from the list
            self.place_objects_from_list(self.world_objects)
        
    def get_possible_goals(self):
        # Get the possible goals for the agent
        possible_goals = []
        for obj in self.world_objects:
            if obj[0] != "agent":
                possible_goals.append(obj[0]+"_" + obj[1])
        return possible_goals


    def place_random_objects(self, num_objects):
        # Place random objects in the grid
        objects = []
        for _ in range(num_objects):
            obj_type = random.choice(["box", "ball", "key"])
            color = random.choice(COLOR_NAMES)
            if obj_type == "box":
                obj = Box(color=color)
            elif obj_type == "ball":
                obj = Ball(color=color)
            elif obj_type == "key":
                obj = Key(color=color)
            
            # Randomly place the object in the grid
            pos = self.place_obj(obj)
            objects.append((obj_type, color, pos))
        return objects

    def place_objects_from_list(self, objects):
        # Place objects in the grid based on the provided list
        for obj_type, color, pos in objects:
            if obj_type == "box":
                obj = Box(color=color)
            elif obj_type == "ball":
                obj = Ball(color=color)
            elif obj_type == "key":
                obj = Key(color=color)
            else:
                continue
            
            # Place the object at the specified position
            self.grid.set(pos[0], pos[1], obj)

    def get_view_exts(self, agent_view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent,
        centered at the agent's position.
        Note: the bottom extent indices are not included in the set.
        If agent_view_size is None, use self.agent_view_size.
        """

        agent_view_size = agent_view_size or self.agent_view_size

        # Calculate the top-left corner of the view extents
        topX = self.agent_pos[0] - agent_view_size // 2
        topY = self.agent_pos[1] - agent_view_size // 2

        # Calculate the bottom-right corner of the view extents
        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY

    
    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)
        
        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = (agent_view_size // 2, agent_view_size // 2)
        if self.carrying:
            grid.set(*agent_pos, self.carrying)            

        return grid, vis_mask

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size // 2)
            - r_vec * (self.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

def main():
    env = PickupOneRoomEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()

    
if __name__ == "__main__":
    main()