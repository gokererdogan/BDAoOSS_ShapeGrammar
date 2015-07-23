# coding=utf-8
'''
Big data analysis of object shape representations
Calculate predictions of the part-based and volumetric model

Created on July 6, 2015

@author: goker erdogan
gokererdogan@gmail.com
https://github.com/gokererdogan/
'''

import BDAoOSS.stimuli.bdaooss_stimuli as bdaooss_stimuli 
import numpy as np
import json
import pandas as pd
import pandasql as pdsql
import pickle
import itertools as it
import BDAoOSS.my_zss as zss # for tree edit distance (https://github.com/timtadh/zhang-shasha)

def __get_depth(t, n):
    """
    Get depth of node n in tree t
    """
    d = 0
    cn = n
    while t[cn].bpointer is not None:
        cn = t[cn].bpointer
        d = d + 1
    return d

def distance_nodes(n1, n2, eq_condition):
    """
    Calculate the distance between two nodes according to given equality conditions.
    n1 and n2 are the two nodes (represented by dictionaries)
    eq_condition is a list of the conditions to take into account. It can contain
        'df': docking face, 'p': position, 's': size, 
        'vold': volume difference, 'voli': 1 - (intersection volume/union volume)
    Returns the distance between nodes.
    """

    # if there are not equality conditions, we just check if the two nodes
    # have the same symbol
    if not eq_condition:
        # if one of the nodes is None (add/remove operation)
        if n1 is None or n2 is None:
            return 1
        if n1['symbol'] != n2['symbol']:
            return 1
    
    # if docking face needs to match for nodes to be equal
    if 'df' in eq_condition:
        if n1 is None or n2 is None:
            return 1
        if n1['df'] != n2['df']: 
            return 1

    # if distance depends on intersection volume 
    if 'voli' in eq_condition:
        if n1 is None:
            p1 = n2['p']
            s1 = np.array([0.,0.,0.])
        else:
            p1 = n1['p']
            s1 = n1['s']
        
        if n2 is None:
            p2 = n1['p']
            s2 = np.array([0.,0.,0.])
        else:
            p2 = n2['p']
            s2 = n2['s']
        
        # bounding box 
        #bb_x0 = min(p1[0] - (s1[0]/2.0), p2[0] - (s2[0]/2.0))
        #bb_x1 = max(p1[0] + (s1[0]/2.0), p2[0] + (s2[0]/2.0))
        #bb_y0 = min(p1[1] - (s1[1]/2.0), p2[1] - (s2[1]/2.0))
        #bb_y1 = max(p1[1] + (s1[1]/2.0), p2[1] + (s2[1]/2.0))
        #bb_z0 = min(p1[2] - (s1[2]/2.0), p2[2] - (s2[2]/2.0))
        #bb_z1 = max(p1[2] + (s1[2]/2.0), p2[2] + (s2[2]/2.0))
        #bb_vol = abs((bb_x0-bb_x1) * (bb_y0-bb_y1) * (bb_z0-bb_z1))
        
        # intersection 
        bb1 = np.array([p1 - (s1/2.0), p1 + (s1/2.0)])
        bb2 = np.array([p2 - (s2/2.0), p2 + (s2/2.0)])
        region = _intersection_boundaries_from_bounding_box(bb1, bb2, 0.0, 0.0)

        # node volumes 
        volp1 = np.abs(np.prod(bb1[1,:] - bb1[0,:]))
        volp2 = np.abs(np.prod(bb2[1,:] - bb2[0,:]))
        
        i_vol = 0
        if region is not None:
            # intersection volume
            i_vol = (np.prod(region[1,:] - region[0,:]))
        # union volume
        u_vol = volp1 + volp2 - i_vol
        return 1 - (i_vol / u_vol)
    
    # if distance is the volume difference 
    if 'vold' in eq_condition:
        if n1 is None:
            p1 = n2['p']
            s1 = np.array([0.,0.,0.])
        else:
            p1 = n1['p']
            s1 = n1['s']
        
        if n2 is None:
            p2 = n1['p']
            s2 = np.array([0.,0.,0.])
        else:
            p2 = n2['p']
            s2 = n2['s']
        
        # intersection 
        bb1 = np.array([p1 - (s1/2.0), p1 + (s1/2.0)])
        bb2 = np.array([p2 - (s2/2.0), p2 + (s2/2.0)])
        region = _intersection_boundaries_from_bounding_box(bb1, bb2, 0.0, 0.0)

        i_vol = 0
        if region is not None:
            i_vol = (np.prod(region[1,:] - region[0,:]))

        # difference = vol(A) + vol(B) - 2*vol(AnB)
        volp1 = np.abs(np.prod(bb1[1,:] - bb1[0,:]))
        volp2 = np.abs(np.prod(bb2[1,:] - bb2[0,:]))
        return volp1 + volp2 - (2 * i_vol)

    
    # if nodes should have the same position to be equal
    if 'p' in eq_condition:
        if n1 is None or n2 is None:
            return 1
        if np.sum(abs(n1['p'] - n2['p'])) > 1e-3:
            return 1

    # if nodes should have the same size to be equal
    if 's' in eq_condition:
        if n1 is None or n2 is None:
            return 1
        if np.sum(abs(n1['s'] - n2['s'])) > 1e-3:
            return 1

    return 0


# We are not looking at subtree kernel because it is hard to interpret what it is doing.

def kernel_tree_edit_distance(self, other, eq_condition):
    """
    Calculate tree edit distance between this and other state
    Because tree edit distance may not be symmetric, we return the 
    average of distances for both ways.
    
    eq_condition is a list of strings specifying the information that needs 
    to be taken into account when checking if two nodes are equal. For now 
    possible choices are:
        'df' for docking face, 'p' for position, 's' for size
        'vold' for volume difference, 'voli' for (1 - (intersection volume/union volume))

    We use the algorithm by 
        Kaizhong Zhang and Dennis Shasha. 
        Simple fast algorithms for the editing distance between trees 
        and related problems. SIAM Journal of Computing, 18:1245–1262, 1989
    We use the implementation from https://github.com/timtadh/zhang-shasha
    NOTE: We made a minor change in zss to allow floating point distances.
    """

    def label_distance(l1, l2):
        """
        Uses distance_nodes function to calculate distance between nodes
        """
        if l1 == '':
            n1 = None
        else:
            n1 = json.loads(l1) 
            n1['p'] = np.array(n1['p'])
            n1['s'] = np.array(n1['s'])

        if l2 == '':
            n2 = None
        else:
            n2 = json.loads(l2) 
            n2['p'] = np.array(n2['p'])
            n2['s'] = np.array(n2['s'])

        dist = distance_nodes(n1, n2, eq_condition)
        return dist 

    # create the trees for each object
    t1 = _kernel_tree_edit_distance_create_tree(self, self.tree.root)
    t2 = _kernel_tree_edit_distance_create_tree(other, other.tree.root)

    return (zss.simple_distance(t1, t2, label_dist=label_distance) + \
            zss.simple_distance(t2, t1, label_dist=label_distance)) / 2.0


def _kernel_tree_edit_distance_create_tree(self, current_node):
    """
    Convert the state to tree format required by zss library for calculating
    tree edit distance
    """

    def get_label(s, n):
        """
        Returns label for a tree node
        label is JSON string of a dictionary containing various info associated
        with the node. 
        """
        t = s.tree
        spatial_state = s.spatial_model.spatial_states[n]
        info = {}
        info['symbol'] = t.nodes[n].tag.symbol
        info['df'] = spatial_state.dock_face
        info['nc'] = len(t.nodes[n].fpointer)
        info['p'] = list(spatial_state.position)
        info['s'] = list(spatial_state.size)
        info['of'] = spatial_state.occupied_faces
        info['d'] = __get_depth(t, n)

        return json.dumps(info) 

    # NOTE: we are not sorting the children. That leads to larger distances
    # if sorting changes the order of nodes in one tree but not the other. 
    # NOTE END

    children = [cn for cn in self.tree.nodes[current_node].fpointer if self.tree.nodes[cn].tag.symbol != 'Null']

    # call create_tree recursively on the children
    t = zss.Node(get_label(self, current_node), children=[_kernel_tree_edit_distance_create_tree(self, cn) for cn in children])
    return t


def similarity_volumetric(obj1, obj2, uncertainty=.1):
    """
    This method calculates the volume of intersection between two objects.
    It can be seen as a measure of distance defined on a volumetric representation
    of shape.
    
    NOTE that this method is quite a simplistic one. It assumes the objects were 
    generated from our shape grammar. 
    In other words, this method probably won't work if these assumptions are violated.
    
    uncertainty is an attempt to introduce position estimation uncertainty into the 
    model. we assume that subjects are uncertain as to the exact location and size
    of a part. That helps in capturing certain cases. See the project notes for one
    such case.

    Right now this method calculates multiple variants of the volumetric similarity 
    idea.
    """
    # intersection and union volumes without uncertainty
    int_vol = _calculate_intersection_volume(obj1, obj2, 0.0, 0.0)
    obj1_vol = _calculate_intersection_volume(obj1, obj1, 0.0, 0.0)
    obj2_vol = _calculate_intersection_volume(obj2, obj2, 0.0, 0.0)
    union_vol = obj1_vol + obj2_vol - int_vol

    # intersection and union volumes with uncertainty
    intu_vol = max(_calculate_intersection_volume(obj1, obj2, uncertainty, 0.0),  _calculate_intersection_volume(obj2, obj1, uncertainty, 0.0))
    obj1u_vol = _calculate_intersection_volume(obj1, obj1, uncertainty, 0.0)
    obj2u_vol = _calculate_intersection_volume(obj2, obj2, uncertainty, 0.0)
    unionu_vol = obj1u_vol + obj2u_vol - intu_vol

    # ----------------------------------------------------------------------------
    # Below is a naive way of calculating volume of intersection. This is not 
    # exactly right because it double counts regions that are the intersection
    # of multiple parts. We just keep this piece of code here with the hope 
    # that it might be useful at some point.
#
#    # find the set of shared parts between objects
#    # we want the parts to have the same position and size
#    i_vol = 0
#    for p1, ss1 in obj1.spatial_model.spatial_states.iteritems():
#        for p2, ss2 in obj2.spatial_model.spatial_states.iteritems():
#                # add to the intersection volume
#                region = _intersection_boundaries_from_spatial_state(ss1, ss2, 0.0, 0.0)
#                if region is not None:
#                    i_vol = i_vol + (np.prod(region[1,:] - region[0,:]))
#
    # ----------------------------------------------------------------------------

    return int_vol, intu_vol, (int_vol/union_vol), (intu_vol/unionu_vol), (int_vol-union_vol), (intu_vol-unionu_vol)


def _calculate_intersection_volume(obj1, obj2, uncertainty1, uncertainty2):
    """
    Calculate the intersection volume between two objects assuming position/size
    uncertainty in either/both objects.
    uncertainty1 is the position/size uncertainty for object 1
    uncertainty2 is the position/size uncertainty for object 2

    This function is the meat of the volumetric similarity model.
    It loops over all possible pairwise intersections making sure that
    we do not double count any regions.
    """
    add_ints = [] # intersections to add
    subt_ints = [] # intersections to subtract
    for ss1 in obj1.spatial_model.spatial_states.values():
        for ss2 in obj2.spatial_model.spatial_states.values():
            # find the intersection between parts ss1 and ss2
            int12 = _intersection_boundaries_from_spatial_state(ss1, ss2, uncertainty1, uncertainty2)
            # if the intersection is not empty, we need to see if it intersects with any region that 
            # we have discovered so far. If it intersects with any region that we will add to the 
            # overall volume, we need to subtract its intersection with the region in question to
            # prevent double counting. Similarly, if it intersects with any region that we will 
            # subtract from the total volume, we need to add the intersection of it with the region.
            if int12 is not None:
                add = []
                subt = []
                add.append(int12)
                for region in add_ints:
                    subt_int = _intersection_boundaries_from_bounding_box(int12, region, 0.0, 0.0)
                    if subt_int is not None:
                        subt.append(subt_int)
                for region in subt_ints:
                    add_int = _intersection_boundaries_from_bounding_box(int12, region, 0.0, 0.0)
                    if add_int is not None:
                        add.append(add_int)

                add_ints.extend(add)
                subt_ints.extend(subt)


    # calculate total intersection volume by adding volumes of regions in add_ints and
    # subtracting the ones in subt_ints
    vol = 0
    for region in add_ints:
        vol = vol + np.prod(region[1,:] - region[0,:])

    for region in subt_ints:
        vol = vol - np.prod(region[1,:] - region[0,:])

    return vol

def _intersection_boundaries_from_spatial_state(ss1, ss2, uncertainty1, uncertainty2):
    """
    Calculate intersection region boundaries from spatial states taking uncertainties
    into account.
    """
    p1 = ss1.position 
    s1 = ss1.size 
    p2 = ss2.position 
    s2 = ss2.size 

    # bounding box of spatial state 1
    bb1 = np.array([p1 - (s1/2.0), p1 + (s1/2.0)])
    # bounding box of spatial state 2
    bb2 = np.array([p2 - (s2/2.0), p2 + (s2/2.0)])
    return _intersection_boundaries_from_bounding_box(bb1, bb2, uncertainty1, uncertainty2)


def _intersection_boundaries_from_bounding_box(p1, p2, uncertainty1, uncertainty2):
    """
    Calculate intersection boundaries from bounding boxes of two parts taking uncertainties
    into account.
    """
    # uncertainy of size/position simply amounts to enlarging the bounding box
    p1s = p1[0,:] - uncertainty1
    p2s = p2[0,:] - uncertainty2
    p1e = p1[1,:] + uncertainty1
    p2e = p2[1,:] + uncertainty2

    # intersection left and right boundaries
    int_start = np.max([p1s, p2s], axis=0)
    int_end = np.min([p1e, p2e], axis=0)
    
    # if all right boundaries (x,y,z) are in fact to the right of the left boundary
    # two parts do intersect.
    if np.all((int_end - int_start) > 0.0):
        return np.array([int_start, int_end])
    return None


 

if __name__ == '__main__':
    ss = pickle.load(open('../stimuli/stimuli20150624_144833/stimuli_set.pkl'))
    predictions = pd.DataFrame(columns=['Comparison1', 'Comparison2'])
    distances = pd.DataFrame(columns=['Transformation'])
    eq_conditions = {'NoCond':[], 'OnlyDF':['df'], 'OnlyP':['p'], 'OnlyS':['s'], \
            'DfP':['df', 'p'], 'DfS':['df', 's'], 'DfPS':['df', 'p', 's'], \
            'VolI':['voli'], 'VolD':['vold']}
                     
    for obj in ss.object_names:
        print obj
        target = ss.stimuli_objects[obj]
        obj_distances = pd.DataFrame(columns=['Transformation'])

        for t in ss.stimuli_names[obj]:
            if t != obj:
                comparison = ss.stimuli_objects[t]

                row = pd.DataFrame({'Transformation':t}, index=[0])

                # tree edit distance ------------------------------------------------------
                # string template for the sql select column expression
                te_col_sql_fmt = '((a.{0:s}<b.{0:s}) + (a.{0:s}<=b.{0:s})) / 2.0 as {0:s}_Prediction, '
                te_col_sql = ''
                # for each eq_condition
                for cond_name, eq_condition in eq_conditions.iteritems():
                    s_te = kernel_tree_edit_distance(target, comparison, eq_condition)
                    col_name = 'TE_' + cond_name
                    row[col_name] = s_te 
                    te_col_sql = te_col_sql + te_col_sql_fmt.format(col_name)

                # volumetric similarity ---------------------------------------------------
                s_voli, s_voliu, s_volir, s_voliru, s_volid, s_voliud = similarity_volumetric(target, comparison)
                row['VOL_Intersection'] = s_voli
                row['VOL_Intersection_Uncertain'] = s_voliu
                row['VOL_IntRatio'] = s_volir
                row['VOL_IntRatio_Uncertain'] = s_voliru
                row['VOL_IntDifference'] = s_volid
                row['VOL_IntDifference_Uncertain'] = s_voliud
                
                # append distances to dataframe
                obj_distances = obj_distances.append(row, ignore_index=True)

        # calculate predictions from each model
        pred_sql = "select a.Transformation as Comparison1, b.Transformation as Comparison2, " + \
                    te_col_sql + \
                    " a.VOL_Intersection>b.VOL_Intersection as VOL_Intersection_Prediction, " + \
                    " a.VOL_Intersection_Uncertain>b.VOL_Intersection_Uncertain as VOL_Intersection_Uncertain_Prediction, " + \
                    " a.VOL_IntRatio>b.VOL_IntRatio as VOL_IntRatio_Prediction, " + \
                    " a.VOL_IntRatio_Uncertain>b.VOL_IntRatio_Uncertain as VOL_IntRatio_Uncertain_Prediction, " + \
                    " a.VOL_IntDifference>b.VOL_IntDifference as VOL_IntDifference_Prediction, " + \
                    " a.VOL_IntDifference_Uncertain>b.VOL_IntDifference_Uncertain as VOL_IntDifference_Uncertain_Prediction " + \
                    " from obj_distances as a, obj_distances as b where a.Transformation!=b.Transformation"

        p = pdsql.sqldf(pred_sql, env=locals())


        distances = distances.append(obj_distances, ignore_index=True)
        predictions = predictions.append(p, ignore_index=True)


    open('ModelPredictions.txt', 'w').write(predictions.to_string())
    open('../../../R/BDAoOSS_Synthetic/ModelPredictions.txt', 'w').write(predictions.to_string())
    open('ModelDistances.txt', 'w').write(distances.to_string())

