'''
Big data analysis of object shape representations
Part-based shape grammar implementation

Created on May 27, 2015

@author: goker erdogan
'''

from AoMRShapeGrammar.pcfg_tree import *
from bdaooss_vision_forward_model import BDAoOSSVisionForwardModel
from AoMRShapeGrammar.shape_grammar import ShapeGrammarState, SpatialModel

"""
Definition of BDAoOSS Probabilistic Context Free Shape Grammar
The only part primitive is a rectangular prism and parts are docked to one 
of the six faces of its parent part. Each P is associated with one part
and `Null` is the terminal end symbol denoting an empty part.
"""
terminals = ['Null']
nonterminals = ['P']
start_symbol = 'P'
rules = {'P' : [['Null'], ['P'], ['P', 'P'], ['P', 'P', 'P']]}
prod_probabilities = {'P' : [.15, .35, .35, .15]}
terminating_rule_ids = {'P' : [0]}

bdaooss_shape_pcfg = PCFG(terminals, nonterminals, start_symbol, rules, prod_probabilities, terminating_rule_ids)

# viewpoint distance. canonical view is from xyz=(5, -5, 5)
VIEWPOINT_RADIUS = np.sqrt(75.0)

# possible docking faces.
FACES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
OPPOSITE_FACES = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4} # the opposite of each face
FACE_COUNT = 6
# used to denote that the part is not docked to any face. e.g., root node
NO_FACE = -1 

class BDAoOSSSpatialState:
    """
    Spatial state class for BDAoOSS shape grammar.
    BDAoOSS spatial model uses this class to hold
    spatial information for nodes (parts) in the
    object.
    we hold the width (x), depth (y), and height (z) 
    of the part, to which face of its parent that 
    the part is docked to and the location of
    its center (xyz),"""
    def __init__(self, size=None, dock_face=None, position=None, occupied_faces=None):
        """
        Initialize spatial state for node
        If any of the parameters are not given, it
        initializes to default values of size 1, docking
        face NO_FACE, and position (0,0,0)
        """
        if size is None or dock_face is None or position is None or occupied_faces is None:
            self.size = np.array([1.5,1,1])
            self.dock_face = NO_FACE
            self.position = np.array([0,0,0])
            self.occupied_faces = []
        else:
            self.size= size
            self.dock_face = dock_face
            self.position = position
            self.occupied_faces = occupied_faces


class BDAoOSSSpatialModel(SpatialModel):
    """
    Spatial Model Class for BDAoOSS Shape Grammar
    For each P node holds the spatial state instance.
    """

    def __init__(self, spatial_states=None):
        """
        Initialize spatial model

        """
        if spatial_states is None:
            self.spatial_states = {}
        else:
            self.spatial_states = spatial_states 

    # -------------------------------------------
    # TO-DO: I don't like passing tree and grammar to below methods, grammar
    # should somehow be accessible to this class already. Think about this later.
    # -------------------------------------------
    def update(self, tree, grammar):
        """
        Updates spatial model, removes nodes that are not in
        nodes parameter and samples sizes, docking faces and
        positions for newly added nodes 
        """
        new_nodes = [node for node in tree.expand_tree(mode=Tree.WIDTH) 
                if tree[node].tag.symbol is not 'Null']
        old_nodes = self.spatial_states.keys()
        removed_nodes = [node for node in old_nodes if node not in new_nodes]
        added_nodes = [node for node in new_nodes if node not in old_nodes]

        for n in removed_nodes:
            del self.spatial_states[n]
        for n in added_nodes:
            if tree[n].bpointer is None: # root node
                parent_sstate = None
            else:
                parent_sstate = self.spatial_states[tree[n].bpointer]

            self.spatial_states[n] = self._get_random_spatial_state(parent_sstate)

    def propose(self, tree, grammar):
        """
        Proposes a new spatial model based on current one.
        Creates a new spatial model with current node states,
        updates it, and returns it
        """    
        sstates_copy = deepcopy(self.spatial_states)
        proposed_spatial_model = BDAoOSSSpatialModel(sstates_copy)
        proposed_spatial_model.update(tree, grammar)
        return proposed_spatial_model

    def _get_random_spatial_state(self, parent_sstate):
        """
        Returns a random spatial state based on the parent
        state's spatial state.
        Samples size, docking face randomly, and calculates
        position.
        """
        # empty constructor creates spatial state for root node.
        new_state = BDAoOSSSpatialState() 
        if parent_sstate is not None:
            # new part is always smaller than its parent in size
            parent_size = parent_sstate.size
            w = np.random.uniform(max(0.1, parent_size[0]-.5), max(0.2, parent_size[0]-.2))
            d = np.random.uniform(max(0.1, parent_size[1]-.5), max(0.2, parent_size[1]-.2))
            h = np.random.uniform(max(0.1, parent_size[2]-.5), max(0.2, parent_size[2]-.2))
            size = np.array([w, d, h])
            # find all empty faces of the parent part
            available_faces = [f for f in range(FACE_COUNT) if f not in parent_sstate.occupied_faces]
            face_ix = np.random.choice(available_faces)
            face = FACES[face_ix] 
            # update parent's occupied faces
            parent_sstate.occupied_faces.append(face_ix)
            # calculate position based on parent's position
            position = parent_sstate.position + (FACES[face_ix,:] * (size + parent_sstate.size) / 2)
            new_state.size = size
            new_state.dock_face = face_ix
            new_state.position = position
            # update this part's occupied faces
            # the opposite of dock_face is now occupied
            new_state.occupied_faces = [OPPOSITE_FACES[face_ix]]

        return new_state


    def probability(self):
        """
        Returns probability of model
        """
        return 0 
    def __str__(self):
        repr_str = "PartName  Size                Position            OccupiedFaces\n"
        fmt = "P         {0:20}{1:20}{2:20}\n"
        for key, state in self.spatial_states.iteritems():
            repr_str = repr_str + fmt.format(np.array_str(state.size, precision=2), 
                        np.array_str(state.position, precision=2), 
                        str(state.occupied_faces)) 
        return repr_str



    
class BDAoOSSShapeState(ShapeGrammarState):
    """
    BDAoOSS shape state class for BDAoOSS grammar and spatial model
    """

    def __init__(self, forward_model=None, data=None, ll_params=None, spatial_model=None, initial_tree=None, viewpoint=None):
        """
        Constructor for BDAoOSSShapeState
        Note that the first parameter ``grammar`` of base class ShapeGrammarState is removed because 
        this class is a grammar specific implementation.
        The additional parameter viewpoint determines the viewing point, i.e., from which point in 3D space we look
        at the object. Here we use spherical coordinates to specify it, i.e., (radius, polar angle, azimuth angle).
        """
        self.MAXIMUM_DEPTH = 2

        if viewpoint is None:
            viewpoint = [VIEWPOINT_RADIUS, 0.0, 0.0]
        self.viewpoint = viewpoint

        ShapeGrammarState.__init__(self, grammar=bdaooss_shape_pcfg, forward_model=forward_model, 
                data=data, ll_params=ll_params, spatial_model=spatial_model, 
                initial_tree=initial_tree)

        # all the other functionality is independent of grammar and spatial model. 
        # hence they are implemented in ShapeGrammarState base class
        # grammar and spatial model specific methods are implemented below

    def convert_to_parts_positions(self):
        """
        Converts the state representation to parts and positions
        representation that can be given to forward model
        """
        parts = []
        positions = []
        scales = []
        for node, state in self.spatial_model.spatial_states.iteritems():
            parts.append(self.tree[node].tag.symbol)
            positions.append(state.position)
            scales.append(state.size)
        return parts, positions, scales, self.viewpoint
    
    
    """The following methods are used for generating stimuli for the experiment."""
    def _stimuli_add_part(self, depth=1):
        """
        Adds a new part to the tree at given depth.
        """
        if depth < 1:
            raise ValueError('Cannot add part to depth<1')
        # if we want to add a part to depth x, we need to find
        # a part at depth x-1
        # get nodes at given depth with available faces
        depth = depth - 1
        tree = self.tree
        sm = self.spatial_model
        depths = {}
        nodes = []
        for node in tree.expand_tree(mode=Tree.WIDTH):
            # if node is a nonterminal and it has no children, it should be expanded
            if tree[node].tag.symbol == 'P': 
                # if root, depth is 0
                if tree[node].bpointer is None:
                    depths[node] = 0
                else:
                    depths[node] = depths[tree[node].bpointer] + 1
                
                cdepth = depths[node]
                # the node should have at most 2 children
                if depths[node] == depth and len(tree[node].fpointer) < 3:
                    nodes.append(node)
                elif depths[node] > depth:
                    break
        
        try:
            node = np.random.choice(nodes)
        except ValueError:
            raise ValueError('No nodes to choose from.')

        # add the new node to tree
        new_node = tree.create_node(tag=ParseNode('P',0), parent=node)
        # add a child Null node
        tree.create_node(tag=ParseNode('Null',''), parent=new_node.identifier)
        # update parent's used production rule
        tree[node].tag.rule = tree[node].tag.rule + 1 
        # update spatial model
        sm.update(tree, bdaooss_shape_pcfg)


    def _stimuli_remove_part(self, depth=1):
        """
        Removes a part from the tree at given depth.
        """
        if depth < 1:
            raise ValueError('Cannot remove part from depth<1')
        

    def _stimuli_vary_part_size(self):
        """Pick a part randomly and change its size.
        """
        tree = self.tree
        nodes = self.spatial_model.spatial_states.keys()
        node = np.random.choice(nodes)
        # max size is parent's size
        pnode = tree[node].bpointer
        max_size = np.array([1.5, 1, 1])
        if pnode is not None:
            max_size = self.spatial_model.spatial_states[pnode].size
        # min size is max of child's size
        cnodes = [n for n in tree[node].fpointer if tree[n].tag.symbol is not 'Null']
        min_size = np.array([.2, .2, .2])
        for cnode in cnodes:
            min_size = np.max(np.vstack([min_size, self.spatial_model.spatial_states[cnode].size]), 0)
        # assign a new random size to part
        self.spatial_model.spatial_states[node].size = np.random.uniform(min_size, max_size)
        # update part's and its children's positions
        for n in tree.expand_tree(nid=node, mode=Tree.WIDTH):
            if tree[n].tag.symbol == 'P':
                nsstate = self.spatial_model.spatial_states[n]
                if tree[n].bpointer is None: # root node
                    nsstate.position = np.array([0, 0, 0])
                else:
                    npsstate = self.spatial_model.spatial_states[tree[n].bpointer]
                    nsstate.position = npsstate.position + (FACES[nsstate.dock_face, :] * (nsstate.size + npsstate.size) / 2) 


    def _stimuli_vary_dock_face(self, depth=1):
        """Pick a part randomly at a given depth and 
        change the face it is docked to.
        Performs the change in place.
        """
        tree = self.tree
        if depth < 1: # can't change the dock_face of root
            raise ValueError('Depth must be greater than 0')
        
        # get nodes at given depth
        depths = {}
        nodes = []
        for node in tree.expand_tree(mode=Tree.WIDTH):
            # if node is a nonterminal and it has no children, it should be expanded
            if tree[node].tag.symbol == 'P': 
                # if root, depth is 0
                if tree[node].bpointer is None:
                    depths[node] = 0
                else:
                    depths[node] = depths[tree[node].bpointer] + 1
                
                cdepth = depths[node]
                if depths[node] == depth:
                    nodes.append(node)
                elif depths[node] > depth:
                    break
        
        try:
            node = np.random.choice(nodes)
        except ValueError:
            raise ValueError('No nodes to choose from. Provided depth > tree depth?')

        parent_node = tree[node].bpointer
        sstate = self.spatial_model.spatial_states[node]
        parent_sstate = self.spatial_model.spatial_states[parent_node]
        # get parent's occupied faces
        pofaces = parent_sstate.occupied_faces
        # available faces are the unoccupied faces of the parent but
        # we need to make sure that when we move the part, it does not
        # clash with one of the child parts of current node.
        # therefore, we remove the opposite faces of occupied faces
        # of our node from the list of available faces too. 
        oofaces = sstate.occupied_faces
        oofaces = [OPPOSITE_FACES[f] for f in oofaces]
        # create a list of the available faces by removing occupied
        # faces
        afaces = [f for f in range(FACE_COUNT) if f not in pofaces and f not in oofaces]
        # pick one randomly from the available faces
        face = np.random.choice(afaces)
        
        # update the occupied_faces of part, set the
        # old dock_face's opposite face to new dock face's
        # opposite
        sstate.occupied_faces[sstate.occupied_faces.index(OPPOSITE_FACES[sstate.dock_face])] = OPPOSITE_FACES[face] 
        
        # update parent's occupied_faces, set the old
        # dock_face to new dock_face
        parent_sstate.occupied_faces[parent_sstate.occupied_faces.index(sstate.dock_face)] = face
        
        # update the dock_face of part
        sstate.dock_face = face

        # update part's and its children's positions
        for n in tree.expand_tree(nid=node, mode=Tree.WIDTH):
            if tree[n].tag.symbol == 'P':
                nsstate = self.spatial_model.spatial_states[n]
                npsstate = self.spatial_model.spatial_states[tree[n].bpointer]
                nsstate.position = npsstate.position + (FACES[nsstate.dock_face, :] * (nsstate.size + npsstate.size) / 2) 



    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        raise NotImplementedError()


    def __neq__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        repr_str = "PartName  Size                Position            OccupiedFaces\n"
        fmt = "P         {0:20}{1:20}{2:20}\n"
        for node in self.tree.expand_tree(mode=Tree.WIDTH):
            if self.tree[node].tag.symbol is not 'Null':
                repr_str = repr_str + fmt.format(np.array_str(self.spatial_model.spatial_states[node].size, precision=2), 
                        np.array_str(self.spatial_model.spatial_states[node].position, precision=2), 
                        str(self.spatial_model.spatial_states[node].occupied_faces)) 

        return repr_str

    def __str__(self):
        return repr(self) 
    
    def __getstate__(self):
        """
        Return data to be pickled. 
        We only need the tree, spatial model, viewpoint, data and ll_params
        """
        state = {}
        state['tree'] = self.tree
        state['spatial_model'] = self.spatial_model
        state['data'] = self.data
        state['ll_params'] = self.ll_params
        state['viewpoint'] = self.viewpoint
        return state
    


if __name__ == '__main__':
    fwdm = BDAoOSSVisionForwardModel()
    data = np.zeros((600, 600))
    params = {'b': 1}

    # randomly generate object
    sm  = BDAoOSSSpatialModel()
    ss = BDAoOSSShapeState(forward_model=fwdm, spatial_model=sm, data=data, ll_params=params)

    print(ss)
    ss.tree.show()
    fwdm._view(ss)
    # fwdm._save_wrl('test.wrl', ss) 
    # fwdm._save_obj('test.obj', ss)
    fwdm._save_stl('test.stl', ss)
    
    """

    # generate object with a specified structure
    t1 = Tree()
    t1.create_node(ParseNode('P', 2), identifier='P1')
    t1.create_node(ParseNode('P', 1), identifier='P11', parent='P1')
    t1.create_node(ParseNode('P', 1), identifier='P12', parent='P1')
    t1.create_node(ParseNode('P', 0), identifier='P111', parent='P11')
    t1.create_node(ParseNode('P', 0), identifier='P121', parent='P12')
    t1.create_node(ParseNode('Null', ''), parent='P111')
    t1.create_node(ParseNode('Null', ''), parent='P121')


    ss1 = {'P1': BDAoOSSSpatialState(size=np.array((1,1,1)), position=np.array((0,0,0)), dock_face=NO_FACE, occupied_faces = [0, 1]), 
            'P11': BDAoOSSSpatialState(size=np.array((.5,.5,.5)), position=np.array((.75, 0, 0)), dock_face=0, occupied_faces = [1,4]), 
            'P12': BDAoOSSSpatialState(size=np.array((.7, .7, .7)), position=np.array((-.85, 0, 0)), dock_face=1, occupied_faces = [0,2]),
            'P111': BDAoOSSSpatialState(size=np.array((.4, .4, .4)), position=np.array((.75, 0, .45)), dock_face=4, occupied_faces = [5]),
            'P121': BDAoOSSSpatialState(size=np.array((.6, .4, .4)), position=np.array((-.85, .55, 0)), dock_face=2, occupied_faces = [3])}

    sm1 = BDAoOSSSpatialModel(ss1)
    ss = BDAoOSSShapeState(forward_model=fwdm, spatial_model=sm1, data=data, ll_params=params, initial_tree=t1)

    print(ss)
    #ss.tree.show()
    fwdm._view(ss)

    ss._stimuli_vary_dock_face(depth=1)
    print(ss)
    fwdm._view(ss)

    ss._stimuli_vary_part_size()
    print(ss)
    #ss.tree.show()
    fwdm._view(ss)

    ss._stimuli_add_part(depth=2)
    print(ss)
    fwdm._view(ss)
    fwdm._save_wrl('test.wrl', ss)
    """    

