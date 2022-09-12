import miney
import pickle
import commands

class MinetestAdapter:
    """
    """
    
    def __init__(self):
        with open('node_dict.pkl', 'rb') as f:
            self.node_dict = pickle.load(f)

        self.segment_dict = {
            0: 'hardenedclay:hardened_clay_white',
            1: 'hardenedclay:hardened_clay_orange',
            2: 'hardenedclay:hardened_clay_magenta',
            3: 'hardenedclay:hardened_clay_light_blue',
            4: 'hardenedclay:hardened_clay_yellow',
            5: 'hardenedclay:hardened_clay_lime',
            6: 'hardenedclay:hardened_clay_pink',
            7: 'hardenedclay:hardened_clay_gray',
            8: 'hardenedclay:hardened_clay_light_gray',
            9: 'hardenedclay:hardened_clay_cyan',
            10: 'hardenedclay:hardened_clay_purple',
            11: 'hardenedclay:hardened_clay_blue',
            12: 'hardenedclay:hardened_clay_brown',
            13: 'hardenedclay:hardened_clay_green',
            14: 'hardenedclay:hardened_clay_red',
            15: 'hardenedclay:hardened_clay_black'
            }

        self.lua_runner = None

    def connect(self, server="127.0.0.1", playername="Minehart", password="", port=29999):
        if self.lua_runner is not None:
            print("Miney already connected")
            return
        self.mt = miney.Minetest(server, playername, password, port)
        self.mt.chat.send_to_all("Hello World")
        self.lua_runner = miney.Lua(self.mt)
        self.lua_runner.run(commands.lock_daytime)

    def build(self, structure, offset=(0,0,0), printMode='segmentation'):
        if self.lua_runner is None:
            print("Miney not connected")
            return
        nodes = []
        dx, dy, dz = offset
        for x in range(len(structure)):
            for y in range(len(structure[x])):
                for z in range(len(structure[x][y])):
                    if structure[x][y][z] != 0:
                        mc_id = structure[x][y][z] % 16
                        if printMode == 'segmentation':
                            mt_name = self.segment_dict[mc_id]
                        elif printMode == 'nodes':
                            try:
                                mt_name = self.node_dict[mc_id]
                            except:
                                mt_name = 'default:stone'
                        else:
                            mt_name = 'default:stone'
                        nodes.append({'x': x+dx, 'y': y+3+dy, 'z': z+dz, 'name': mt_name})

        lua = ''
        for node in nodes:
            if node["name"] != "ignore":
                lua = lua + f"minetest.set_node(" \
                            f"{self.lua_runner.dumps({'x': node['x'], 'y': node['y'], 'z': node['z']})}, " \
                            f"{{name=\"{node['name']}\"}})\n"
                                                
        self.lua_runner.run(lua)
