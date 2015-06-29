"""
Stimuli classes for BDAoOSS experiment.
These classes are used for storing the stimuli to disk.
We don't want to store the BDAoOSSShapeState instances 
directly to disk because these objects tend to take a
large amount of space.

11 June 2015
gokererdogan@gmail.com
https://github.com/gokererdogan/
Goker Erdogan
"""


class BDAoOSSStimuliSet:
    """
    Class for storing the stimulus set used in the experiments.
    """
    def __init__(self, grammar, created_on, object_count, object_names, stimuli_names, stimuli_objects):
        """
        grammar: PCFG grammar used to generate the stimuli
	created_on: A string containing the creation time of the stimuli set
        object_count: number of distinct objects in the stimulus set.
            base objects are named oX, i.e., o1, o2, o3 and so on.
        object_names: The names of generated objects.
        stimuli_names: The names of generated stimuli. A dictionary with object_name
            as key. stimuli_names['o1'] = list of stimuli names for o1.
            See generate_stimuli.py for stimulus name format.
        stimuli_objects: BDAoOSSStimuli instances for each stimuli. This is a 
            dictionary with stimuli names as key.
        """
        self.grammar = grammar
        self.object_count = object_count
        self.object_names = object_names
        self.stimuli_names = stimuli_names
        self.stimuli_objects = stimuli_objects
        self.created_on = created_on 

class BDAoOSSStimuli:
    """
    Class for storing a single stimuli (i.e., object generated
    from the shape grammar).
    Contains only the necessary data to re-create the 
    BDAoOSSShapeState.
    """
    def __init__(self, name, tree, spatial_model):
        """
        name: Name of the stimulus. see generate_stimuli.py for naming
            conventions.
        tree: The parse tree for the object
        spatial_model: BDAoSSSpatialModel instance containing the 
            spatial model for the object.
        """
        self.name = name
        self.tree = tree
        self.spatial_model = spatial_model



