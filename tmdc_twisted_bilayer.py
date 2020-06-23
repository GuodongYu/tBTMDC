import numpy as np
from pymatgen.core.structure import Structure as pmg_struct
from scipy.linalg.lapack import zheev
from tBG.fortran.spec_func import get_pk
from tBG.utils import *
from tBG.hopping import filter_neig_list
import copy
from kagome_twisted_bilayer  import pmg_sublatts_twisted_bilayer_hexagonal_lattice

#### inter layer hopping ########################
inter_hop_params = {'S-S':{'sigma':{'v_b':2.627, 'R_b':3.128, 'eta_b':3.859},
                           'pi':{'v_b':-0.708, 'R_b':2.923, 'eta_b':5.724}},
                    'Se-Se':{'sigma':{'v_b':2.559, 'R_b':3.337, 'eta_b':4.114},
                             'pi':{'v_b':-1.006, 'R_b':2.927, 'eta_b':5.185}}}

def get_local_axes(vec):
    vec0 = np.array(vec)
    vec1 = np.array([0, -vec[2],vec[1]])
    vec2 = np.cross(vec0, vec1)
    return np.array([vec0/np.linalg.norm(vec0), 
                     vec1/np.linalg.norm(vec1),
                     vec2/np.linalg.norm(vec2)])

def get_direct_cosine_with_local_axes(vec, local_axes):
    vec_unit = np.array(vec)/np.linalg.norm(vec)
    cos_0 = np.dot(vec_unit, local_axes[0])
    cos_1 = np.dot(vec_unit, local_axes[1])
    cos_2 = np.dot(vec_unit, local_axes[2])
    return np.array([cos_0, cos_1, cos_2])

def orb_vec(orb, ref_x):
    if orb=='px':
        return np.append(ref_x,[0])
    elif orb=='py':
        vec=rotate_on_vec(90, ref_x)
        return np.append(vec, [0])
    elif orb=='pz':
        return np.array([0,0,1])

def Vpp_sigma_pi(hop_param):
    v_b = hop_param['v_b']
    R_b = hop_param['R_b']
    eta_b = hop_param['eta_b']
    def f(r):
        return v_b*np.exp(-(r/R_b)**eta_b)
    return f

class InterHop:
    def __init__(self, neigh_pair):
        self.Vpp_sigma_func = Vpp_sigma_pi(inter_hop_params[neigh_pair]['sigma'])        
        self.Vpp_pi_func = Vpp_sigma_pi(inter_hop_params[neigh_pair]['pi'])        
        self.orbs = ['pzo','pxo','pyo','pze','pxe','pye']

    def get_inter_hop(self, orb0, ref_x_0, orb1, ref_x_1, dr_vec):
        vec0 = orb_vec(orb0, ref_x_0)
        vec1 = orb_vec(orb1, ref_x_1)
        local_axes = get_local_axes(dr_vec)
        dirt_cos_0 = get_direct_cosine_with_local_axes(vec0, local_axes)
        dirt_cos_1 = get_direct_cosine_with_local_axes(vec1, local_axes)
        Vpp_pi = self.Vpp_pi_func(np.linalg.norm(dr_vec))
        Vpp_sigma = self.Vpp_sigma_func(np.linalg.norm(dr_vec))
        return np.sum(np.array(dirt_cos_0)*np.array(dirt_cos_1)*np.array([Vpp_sigma, Vpp_pi, Vpp_pi]))

    def get_inter_hop_mat(self, dr_vec, spin, ref_x_0, ref_x_1):
        t_mat = np.zeros((len(self.orbs)*spin, len(self.orbs)*spin))
        for ind0 in range(len(self.orbs)):
            orb0 = self.orbs[ind0]
            c0 = 1/np.sqrt(2)
            for ind1 in range(len(self.orbs)):
                orb1 = self.orbs[ind1]
                if orb1[-1]=='o':
                    c1 = 1/np.sqrt(2)
                elif orb1[-1]=='e':
                    c1 = -1/np.sqrt(2)
                t = c0*c1*self.get_inter_hop(orb0[:-1], ref_x_0, orb1[:-1], ref_x_1, -dr_vec) 
                t_mat[ind0*spin:(ind0+1)*spin, ind1*spin:(ind1+1)*spin] = t*np.identity(spin)
        return t_mat
        
    
def inter_hop_pz(r, h, neigh_pair):
    n = h/r
    Vpp_pi = Vpp_sigma_pi(inter_hop_params[neigh_pair]['pi'])(r)
    Vpp_sigma = Vpp_sigma_pi(inter_hop_params[neigh_pair]['sigma'])(r)
    t = n**2*Vpp_sigma+(1-n**2)*Vpp_pi
    return t

def inter_hop(r, h, neigh_pair, orb_pair):
    if orb_pair =='pzo-pzo':
        return 1/2*inter_hop_pz(r, h, neigh_pair)
    elif orb_pair == 'pzo-pze':
        return -1/2*inter_hop_pz(r, h, neigh_pair)
    elif orb_pair == 'pze-pzo':
        return 1/2*inter_hop_pz(r, h, neigh_pair)
    elif orb_pair == 'pze-pze':
        return -1/2*inter_hop_pz(r, h, neigh_pair)
    else:
        return np.zeros(len(r))

def inter_hop_mat(r, h, neigh_pair, spin):
    orbs = ['pzo','pxo','pyo','pze','pxe','pye']
    t_mat = np.array([np.zeros((len(orbs)*spin, len(orbs)*spin))]*len(r))
    for ind0 in range(len(orbs)):
        orb0 = orbs[ind0]
        for ind1 in range(len(orbs)):
            orb1 = orbs[ind1]
            orb_pair = orb0+'-'+orb1
            t = inter_hop(r,h, neigh_pair, orb_pair) 
            t_mat[:, ind0*spin:(ind0+1)*spin, ind1*spin:(ind1+1)*spin] = np.array([i*np.identity(spin) for i in t])
    return t_mat
            
##################################################

###### SOC params##########################
# to do
##########################################


##### intra layer hopping ######################################
## MoS2##
hop_params_intra_MoS2=\
     {'onsite':
           {'M':{'dxz':1.0688, 'dyz':1.0688, 'dz2':-0.1380, 'dxy':0.0874, 'dx2-y2':0.0874}, 
            'X':{'pzo':-0.7755, 'pxo':-1.2902, 'pyo':-1.2902, 'pze':-2.8949, 'pxe':-1.9065, 'pye':-1.9065}},
      'hopping':
           {'M-M':{'dxz-dxz':-0.2069, 'dyz-dyz':0.0323,'dz2-dz2':-0.2979,'dxy-dxy':0.2747, 'dx2-y2-dx2-y2':-0.5581,\
                   'dz2-dx2-y2':0.4096, 'dxz-dyz':-0.2562, 'dz2-dxy':-0.1145, 'dxy-dx2-y2':-0.2487}, 
            'X-X':{'pzo-pzo':-0.1739, 'pxo-pxo':0.8651, 'pyo-pyo':-0.1872, 'pze-pze':-0.1916, 'pxe-pxe':0.9122,\
                   'pye-pye':0.0059, 'pzo-pyo':-0.0679, 'pze-pye':0.0075, 'pzo-pxo':-0.0995, 'pxo-pyo':-0.0705,\
                   'pze-pxe':0.1063, 'pxe-pye':-0.0385},
            'X-M':{'pxo-dxz':-0.7883, 'pzo-dyz':-1.3790, 'pyo-dyz':2.1584, 'pze-dz2':-0.8836, 'pye-dz2':-0.9402,\
                   'pxe-dxy':1.4114, 'pze-dx2-y2':-0.9535, 'pye-dx2-y2':0.6517},
            'X-M-2nd':{'pze-dz2':-0.0686, 'pye-dz2':-0.1498, 'pze-dx2-y2':-0.2205, 'pye-dx2-y2':-0.2451}}}

## MoSe2 ##
###################################################################################################################


class IntraHopExpand:
    """
    recover all paramters
    """
    def __init__(self, hop_params_intra):
        self.hop_params_intra = hop_params_intra
        ## the orbital order is the same as Shiang's paper prb 92 205108 [2015])
        self.orbs = ['dxz','dyz','pzo','pxo','pyo','dz2','dxy','dx2-y2','pze','pxe','pye']
        self.orb_ind = dict(zip(self.orbs, range(len(self.orbs))))
    
    def hop_value(self, orb0, orb1, angle, neig_type=''):
        orb_pair = orb0 + '-' + orb1

        if neig_type=='X-M-2nd':
            return self.hop_class6(orb0, orb1, angle, neig_type)

        if orb0 == orb1:
            return self.hop_class1(orb0, angle, neig_type)
    
        elif orb_pair in ['pzo-pyo','dz2-dx2-y2','pze-pye']:
            return self.hop_class23(orb0, orb1, angle, neig_type, 2)
        elif orb_pair in ['pyo-pzo','dx2-y2-dz2','pye-pze']:
            if angle in [0,180]:
                return self.hop_value(orb1, orb0, 180-angle, neig_type)
            elif angle in [60, 120]:
                return self.hop_value(orb1, orb0, angle-180, neig_type)
            elif angle in [-60, -120, -180]:
                return self.hop_value(orb1, orb0, angle+180, neig_type)
                
    
        elif orb_pair in ['dxz-dyz','pzo-pxo','pxo-pyo','dz2-dxy','dxy-dx2-y2','pze-pxe','pxe-pye']:
            return self.hop_class23(orb0, orb1, angle, neig_type, 3)
        elif orb_pair in ['dyz-dxz','pxo-pzo','pyo-pxo','dxy-dz2','dx2-y2-dxy','pxe-pze','pye-pxe']:
            if angle in [0,180]:
                return self.hop_value(orb1, orb0, 180-angle, neig_type)
            elif angle in [60, 120]:
                return self.hop_value(orb1, orb0, angle-180, neig_type)
            elif angle in [-60, -120, -180]:
                return self.hop_value(orb1, orb0, angle+180, neig_type)
    
        elif orb_pair in ['pzo-dxz','pyo-dxz','pxo-dyz','pxe-dz2','pze-dxy','pye-dxy','pxe-dx2-y2']:
            return self.hop_class45(orb0, orb1, angle, neig_type, 4)
    
        elif orb_pair in ['pxo-dxz','pzo-dyz','pyo-dyz','pze-dz2','pye-dz2','pxe-dxy','pze-dx2-y2','pye-dx2-y2']:
            return self.hop_class45(orb0, orb1, angle, neig_type, 5)

        else:
            return 0.
    
    def hop_class1(self, orb, angle, neigh_type):
        orb_pair = orb+'-'+orb
        tii_1 = self.hop_params_intra['hopping'][neigh_type][orb_pair]
        ind = self.orb_ind[orb]
        if ind in np.array([1,4,7,10])-1: 
            # ind: alpha
            orb_beta = self.orbs[ind+1]
            orb_pair_beta_beta= orb_beta+'-'+orb_beta
            t_beta_beta_1 = self.hop_params_intra['hopping'][neigh_type][orb_pair_beta_beta]
            tii_2 = 1/4*tii_1 + 3/4*t_beta_beta_1
        elif ind in np.array([3,6,9])-1: 
            # ind: gamma
            tii_2 = tii_1
        elif ind in np.array([2,5,8,11])-1:
            # ind: beta
            orb_alpha = self.orbs[ind-1]
            orb_pair_alpha_alpha = orb_alpha + '-' + orb_alpha
            t_alpha_alpha_1 = self.hop_params_intra['hopping'][neigh_type][orb_pair_alpha_alpha]
            tii_2 = 1/4*tii_1+3/4*t_alpha_alpha_1
        if angle in [0, 180, -180]:
            return tii_1
        elif angle in [60, 120, -60, -120]:
            return tii_2
        else:
            print(angle)
            raise ValueError()
    
    def hop_class23(self, orb0, orb1, angle, neigh_type, class_type):
        orb_pair = orb0+'-'+orb1
        tij_1 = self.hop_params_intra['hopping'][neigh_type][orb_pair]
        ind0 = self.orb_ind[orb0]
        ind1 = self.orb_ind[orb1]
        if [ind0, ind1] in [list(i) for i in np.array([[3,5],[6,8],[9,11]])-1]:
            # orb0:gamma orb1:beta
            orb_alpha = self.orbs[ind1-1]
            orb_pair_gamma_alpha = orb0 +'-'+orb_alpha
            t_gamma_alpha_1 = self.hop_params_intra['hopping'][neigh_type][orb_pair_gamma_alpha]
            tij_2 = np.sqrt(3)/2*t_gamma_alpha_1 - 1/2*tij_1
            tij_3 = -np.sqrt(3)/2*t_gamma_alpha_1 - 1/2*tij_1
        elif [ind0, ind1] in [list(i) for i in np.array([[1,2],[4,5],[7,8],[10,11]])-1]: 
            # orb0:alpha orb1:beta
            orb_pair_alpha_alpha = orb0+'-'+orb0 
            orb_pair_beta_beta = orb1+'-'+orb1
            t_alpha_alpha_1 = self.hop_params_intra['hopping'][neigh_type][orb_pair_alpha_alpha]
            t_beta_beta_1 = self.hop_params_intra['hopping'][neigh_type][orb_pair_beta_beta]
            tij_2 = np.sqrt(3)/4*(t_alpha_alpha_1-t_beta_beta_1)-tij_1
            tij_3 = -np.sqrt(3)/4*(t_alpha_alpha_1-t_beta_beta_1)-tij_1
        elif [ind0, ind1] in [list(i) for i in np.array([[3,4],[6,7],[9,10]])-1]:
            # orb0:gamma orb1:alpha
            orb_beta = self.orbs[ind1+1]
            orb_pair_gamma_beta = orb0+'-'+orb_beta
            t_gamma_beta_1 = self.hop_params_intra['hopping'][neigh_type][orb_pair_gamma_beta]
            tij_2 = np.sqrt(3)/2*t_gamma_beta_1 + 1/2.*tij_1
            tij_3 = -np.sqrt(3)/2*t_gamma_beta_1 + 1/2.*tij_1
        if class_type==2:
            if angle in [0, 180, -180]:
                return tij_1
            elif angle in [60, 120]:
                return tij_3
            elif angle in [-60, -120]:
                return tij_2
            else:
                raise ValueError()
        elif class_type==3:
            if angle == 0:
                return -tij_1
            elif angle in [180, -180]:
                return tij_1
            elif angle == 60:
                return -tij_3
            elif angle == 120:
                return tij_3
            elif angle ==-60:
                return -tij_2
            elif angle == -120:
                return tij_2
            else:
                print(angle)
                raise ValueError()
    
    def hop_class45(self, orb0, orb1, angle, neigh_type, class_type):
        orb_pair = orb0+'-'+orb1
        try:
            tij_5 = self.hop_params_intra['hopping'][neigh_type][orb_pair]
        except:
            pass
        ind0 = self.orb_ind[orb0]
        ind1 = self.orb_ind[orb1]
        if [ind0, ind1] in [list(i) for i in np.array([[4,1],[10,7]])-1]:
            #ind0:alphap ind1:alpha
            orb_betap = self.orbs[ind0+1]
            orb_beta = self.orbs[ind1+1]
            orb_pair_betap_beta = orb_betap+'-'+orb_beta
            t_betap_beta_5 = self.hop_params_intra['hopping'][neigh_type][orb_pair_betap_beta]
            tij_4 = 1/4*tij_5 + 3/4*t_betap_beta_5
        elif [ind0, ind1] in [list(i) for i in np.array([[5,2],[11,8]])-1]:
            # ind0:betap ind1:beta
            orb_alphap = self.orbs[ind0-1]
            orb_alpha = self.orbs[ind1-1]
            orb_pair_alphap_alpha = orb_alphap + '-' + orb_alpha
            t_alphap_alpha_5 = self.hop_params_intra['hopping'][neigh_type][orb_pair_alphap_alpha]
            tij_4 = 3/4*t_alphap_alpha_5 + 1/4*tij_5

        elif [ind0, ind1] in [list(i) for i in np.array([[5,1],[11,7]])-1]:
            # ind0:betap ind1:alpha
            orb_alphap = self.orbs[ind0-1]
            orb_beta = self.orbs[ind1+1]
            orb_pair_alphap_alpha = orb_alphap+'-'+orb1
            orb_pair_betap_beta = orb0+'-'+orb_beta
            t_alphap_alpha_5 = self.hop_params_intra['hopping'][neigh_type][orb_pair_alphap_alpha]
            t_betap_beta_5 = self.hop_params_intra['hopping'][neigh_type][orb_pair_betap_beta]
            tij_4 = -np.sqrt(3)/4*t_alphap_alpha_5 + np.sqrt(3)/4*t_betap_beta_5

        elif [ind0, ind1] in [list(i) for i in np.array([[4,2],[10,8]])-1]:
            #ind0:alphap, ind1:beta
            orb_alpha = self.orbs[ind1-1]
            orb_betap = self.orbs[ind0+1]
            orb_pair_alphap_alpha = orb0+'-'+orb_alpha
            orb_pair_betap_beta = orb_betap+'-'+orb1
            t_alphap_alpha_5 = self.hop_params_intra['hopping'][neigh_type][orb_pair_alphap_alpha]
            t_betap_beta_5 = self.hop_params_intra['hopping'][neigh_type][orb_pair_betap_beta]
            tij_4 = -np.sqrt(3)/4*t_alphap_alpha_5 + np.sqrt(3)/4*t_betap_beta_5
            
        elif [ind0, ind1] in [list(i) for i in np.array([[3,1],[9,7]])-1]:
            #ind0:gammap ind1:alpha
            orb_beta = self.orbs[ind1+1]
            orb_pair_gammap_beta = orb0+'-'+orb_beta
            t_gammap_beta_5 = self.hop_params_intra['hopping'][neigh_type][orb_pair_gammap_beta]
            tij_4 = -np.sqrt(3)/2*t_gammap_beta_5
        elif [ind0, ind1] in [list(i) for i in np.array([[3,2],[9,8]])-1]:
            tij_4 = -1/2*tij_5
        elif ind0==9-1 and ind1==6-1:
            tij_4 = tij_5
        elif ind0==10-1 and ind1==6-1:
            orb11 = self.orbs[10]
            orb6 = self.orbs[5]
            orb_pair_11_6 = orb11+'-'+orb6
            t11_6_5 = self.hop_params_intra['hopping'][neigh_type][orb_pair_11_6]
            tij_4 = -np.sqrt(3)/2*t11_6_5
        elif ind0==11-1 and ind1==6-1:
            tij_4 = -1/2*tij_5

        if class_type==4:
            if angle==90:
                return 0
            elif angle==-30:
                return -tij_4
            elif angle==-150:
                return tij_4
            else:
                print(angle)
                raise ValueError()
        elif class_type==5:
            if angle==90:
                return tij_5
            elif angle in [-30,-150]:
                return tij_4
            else:
                print(angle)
                raise ValueError()
    
    def hop_class6(self, orb0, orb1, angle, neigh_type):
        orb_pair = orb0 + '-' + orb1
        ind0 = self.orb_ind[orb0]
        ind1 = self.orb_ind[orb1]
        if ind0==9-1 and ind1==6-1:
            t96_6 = self.hop_params_intra['hopping'][neigh_type][orb_pair]
            if angle in [30, 150, -90]:
                return t96_6
        elif ind0==11-1 and ind1==6-1:
            t116_6 = self.hop_params_intra['hopping'][neigh_type][orb_pair]
            if angle in [30, 150]:
                return -1/2*t116_6
            elif angle==-90:
                return t116_6
        elif ind0==10-1 and ind1==6-1:
            orb_pair_11_6 = self.orbs[10]+'-'+self.orbs[5]
            t116_6 =  self.hop_params_intra['hopping'][neigh_type][orb_pair_11_6]
            t106_6 = np.sqrt(3)/2*t116_6
            if angle==30:
                return -t106_6
            elif angle==150:
                return t106_6
            elif angle==-90:
                return 0
        elif ind0==9-1 and ind1==8-1:
            t98_6 = self.hop_params_intra['hopping'][neigh_type][orb_pair]
            if angle==-90:
                return t98_6
            elif angle in [30, 150]:
                return -1/2*t98_6
        elif ind0==9-1 and ind1==7-1:
            orb_pair_9_8 = self.orbs[8]+'-'+self.orbs[7]
            t98_6 = self.hop_params_intra['hopping'][neigh_type][orb_pair_9_8]
            t97_6 = np.sqrt(3)/2*t98_6
            if angle==30:
                return -t97_6
            elif angle==150:
                return t97_6
            elif angle==-90:
                return 0
        elif ind0==10-1 and ind1==7-1:
            orb_pair_11_8 = self.orbs[10]+'-'+self.orbs[7]
            t118_6 = self.hop_params_intra['hopping'][neigh_type][orb_pair_11_8]
            t107_6 = 3/4*t118_6
            if angle in [30, 150]:
                return t107_6
            elif angle==-90:
                return 0
        elif [ind0, ind1] in [[11-1, 7-1],[10-1, 8-1]]:
            orb_pair_11_8 = self.orbs[10]+'-'+self.orbs[7]
            t118_6 = self.hop_params_intra['hopping'][neigh_type][orb_pair_11_8]
            tij_6 = np.sqrt(3)/4*t118_6
            if angle==30:
                return tij_6
            elif angle==150:
                return -tij_6
            elif angle==-90:
                return 0
        elif ind0==11-1 and ind1==8-1:
            t118_6 = self.hop_params_intra['hopping'][neigh_type][orb_pair]
            if angle==-90:
                return t118_6
            elif angle in [30, 150]:
                return 1/4*t118_6
        else:
            return 0
                
class IntraHopMat:
    def __init__(self, hop_params_intra, spin):
        ## this order is for Hamiltonian
        self.orbs={'M':['dxz','dyz','dz2','dxy','dx2-y2'],
                   'X':['pzo','pxo','pyo','pze','pxe','pye']}
        self.spin = spin
        self.hop_params_intra = hop_params_intra
        self.intra_hop_expand = IntraHopExpand(hop_params_intra)
        self.onsite_mat = self._onsite_mat()

    def _onsite_mat(self):
        out = {}
        for tp in ['M','X']:
            orbs_tp = self.orbs[tp]
            e_on = np.zeros((len(orbs_tp)*self.spin, len(orbs_tp)*self.spin))
            for ith_orb in range(len(orbs_tp)):
                orb = orbs_tp[ith_orb]
                e_on[ith_orb*self.spin:(ith_orb+1)*self.spin, ith_orb*self.spin:(ith_orb+1)*self.spin] = \
                                        self.hop_params_intra['onsite'][tp][orb]*np.identity(self.spin)
            out[tp] = e_on
        return out
    
    def get_hop_mat(self, neigh_type, angle):
        site0 = neigh_type.split('-')[0]
        site1 = neigh_type.split('-')[1]
        t_mat = np.zeros((len(self.orbs[site0])*self.spin, len(self.orbs[site1])*self.spin))
        for ind0 in range(len(self.orbs[site0])):
            orb0 = self.orbs[site0][ind0]
            for ind1 in range(len(self.orbs[site1])):
                orb1 = self.orbs[site1][ind1]
                t = self.intra_hop_expand.hop_value(orb0, orb1, angle, neigh_type)
                t_mat[ind0*self.spin:(ind0+1)*self.spin, ind1*self.spin:(ind1+1)*self.spin] = t*np.identity(self.spin)
        return t_mat



def get_neighbors_MX2(pmg_struct, dist_cut, layer_nsites):

    neig_list = pmg_struct.get_neighbor_list(dist_cut)
    p0, p1, offset, dist = neig_list
    nsite = np.sum(layer_nsites)
    neigh_intra = {'bottom':{'M-M':[],'X-X':[], 'X-M':[], 'X-M-2nd':[]},\
                           'top':{'M-M':[],'X-X':[], 'X-M':[], 'X-M-2nd':[]}}

    def get_neigh_intra_pair(neigh_list, nth_nearest):
        a, b, c, d = neigh_list
        d = np.float16(d)
        dists = np.unique(d)
        c = np.array(c[:,0:2], dtype=int)
        out = []
        for i in range(nth_nearest):
            ind = np.where(d==dists[i])[0]
            ai = a[ind]
            bi = np.append(c[ind], b[ind].reshape(-1,1), axis=1)
            out.append(np.append(ai.reshape(-1,1), bi, axis=1))
        return out


    def add_neigs_intra(layer, neigh_pair):
        if layer=='bottom':
            if neigh_pair=='M-M':
                ind0 = np.where(p0<=int(layer_nsites[0]/2)-1)[0]
                ind1 = np.where(p1<=int(layer_nsites[0]/2)-1)[0]
            elif neigh_pair=='X-X':
                ind0 = np.intersect1d(np.where(p0>=int(layer_nsites[0]/2))[0],np.where(p0<=layer_nsites[0]-1)[0])
                ind1 = np.intersect1d(np.where(p1>=int(layer_nsites[0]/2))[0],np.where(p1<=layer_nsites[0]-1)[0])
            elif neigh_pair=='X-M':
                ind1 = np.where(p1<=int(layer_nsites[0]/2)-1)[0]
                ind0 = np.intersect1d(np.where(p0>=int(layer_nsites[0]/2))[0],np.where(p0<=layer_nsites[0]-1)[0])
        elif layer=='top':
            if neigh_pair=='M-M':
                ind0 = np.intersect1d(np.where(p0>=layer_nsites[0])[0],\
                                      np.where(p0<=layer_nsites[0]+int(layer_nsites[1]/2)-1)[0])
                ind1 = np.intersect1d(np.where(p1>=layer_nsites[0])[0],\
                                      np.where(p1<=layer_nsites[0]+int(layer_nsites[1]/2)-1)[0])
            elif neigh_pair=='X-X':
                ind0 = np.where(p0>=layer_nsites[0]+layer_nsites[1]/2)[0]
                ind1 = np.where(p1>=layer_nsites[0]+layer_nsites[1]/2)[0]
            elif neigh_pair=='X-M':
                ind1 = np.intersect1d(np.where(p1>=layer_nsites[0])[0],\
                                      np.where(p1<=layer_nsites[0]+int(layer_nsites[1]/2)-1)[0]) 
                ind0 = np.where(p0>=layer_nsites[0]+layer_nsites[1]/2)[0]
        ind = np.intersect1d(ind0, ind1)
        if neigh_pair in ['M-M','X-X']:
            neigh_list_cut = filter_neig_list([p0[ind], p1[ind], offset[ind],dist[ind]])
        else:
            neigh_list_cut = [p0[ind], p1[ind], offset[ind],dist[ind]]

        if neigh_pair == 'X-M':
            neigh_intra[layer]['X-M'], neigh_intra[layer]['X-M-2nd'] = \
                         get_neigh_intra_pair(neigh_list_cut, 2)
        else:
            neigh_intra[layer][neigh_pair] =  get_neigh_intra_pair(neigh_list_cut, 1)[0]

    for layer in ['bottom', 'top']:
        try:
            for neigh_pair in ['M-M', 'X-X', 'X-M']:
                add_neigs_intra(layer, neigh_pair)
        except:
            pass

    neigh_inter = []
    def get_neigs_inter():
        ind0 = np.intersect1d(np.where(p0>=int(layer_nsites[0]/2))[0],np.where(p0<=layer_nsites[0]-1)[0])
        ind1 = np.where(p1>=layer_nsites[0]+layer_nsites[1]/2)[0]
        ind = np.intersect1d(ind0, ind1)
        p0i = p0[ind]
        p1i = p1[ind]
        offseti = np.array(offset[ind][:,0:2], dtype=int)
        p1i_extend = np.append(offseti, p1i.reshape(-1,1), axis=1)
        disti = dist[ind]
        return [p0i, p1i_extend, disti]
    try:
        neigh_inter = get_neigs_inter()
    except:
        pass
    return neigh_intra, neigh_inter

class _PBCMethods:
    """
    methods for twisted bilayer kagome lattice
    """
    def pymatgen_structure(self):
        return pmg_struct(self.latt_vec, ['C']*len(self.coords), self.coords, coords_are_cartesian=True)


    def hamilt_cell_diff(self, k, elec_field=0.0):
        Hk = np.zeros((self.nsite, self.nsite),dtype=complex)
        if elec_field:
            np.fill_diagonal(Hk,self.coords[:,-1]*elec_field)
        latt_vec = self.latt_vec[0:2][:,0:2]
        for i in range(self.nsite):
            for m,n,j in self.hoppings[i]:
                R = m*latt_vec[0]+n*latt_vec[1]
                t = self.hoppings[i][(m,n,j)]
                phase = np.exp(1j*np.dot(k, R))
                Hk[i,j] = Hk[i,j] + t*phase
                Hk[j,i] = Hk[j,i] + t*np.conj(phase)
        return Hk

    def hamilt_pos_diff(self, k, elec_field=0.0):
        if len(k) == 2:
            k = np.array([k[0],k[1],0.])
        
        Hk = np.zeros((self.norb*self.spin, self.norb*self.spin),dtype=complex)
        if elec_field:
            np.fill_diagonal(Hk,self.coords[:,-1]*elec_field)
            e_on_elec = self.coords[:,-1]*elec_field
        latt_vec = self.latt_vec[0:2]
        for i in range(self.nsite):
            ri = self.coords[i]
            for m,n,j in self.hoppings[i]:
                R = m*latt_vec[0]+n*latt_vec[1]
                rj = R + self.coords[j]
                t = self.hoppings[i][(m,n,j)]
                phase = np.exp(1j*np.dot(k, rj-ri))
                Hk[self.site_slice[i],self.site_slice[j]] = \
                      Hk[self.site_slice[i],self.site_slice[j]] + t*phase
                Hk[self.site_slice[j],self.site_slice[i]] = \
                      Hk[self.site_slice[j],self.site_slice[i]] + t.T*np.conj(phase)
        for i in range(self.nsite):
            e_on = self.onsite_energy[i]
            Hk[self.site_slice[i],self.site_slice[i]] = \
                  Hk[self.site_slice[i],self.site_slice[i]] + e_on
        return Hk

    def diag_kpts(self, kpts, vec=0, pmk=0, elec_field=0.):
        """
        kpts: the coordinates of kpoints
        vec: whether to calculate the eigen vectors
        pmk: whether to calculate PMK for effective band structure
        elec_field: the electric field perpendicular to graphane plane
        fname: the file saveing results
        """
        val_out = []
        vec_out = []
        pmk_out = []
        for k in kpts:
            #Hk = self.hamilt_cell_diff(k, elec_field)
            Hk = self.hamilt_pos_diff(k, elec_field)
            vals, vecs, info = zheev(Hk, vec)
            if info:
                raise ValueError('zheev failed')
            if pmk:
                Pk = get_pk(k, np.array(self.layer_nsites)/2, [1,1], 2, 2, vecs, self.coords, self.species())
                pmk_out.append(Pk)
            val_out.append(vals)
            vec_out.append(vecs)
        return np.array(val_out), np.array(vec_out), np.array(pmk_out)

class _MX2TB_Methods:
    """
    methods for twisted bilayer kagome lattice
    """
    def pymatgen_structure(self):
        try:
            ### for two layers ###
            n_bott, n_top = self.layer_nsites
            eles = ['Mo']*int(n_bott/2) + ['S']*int(n_bott/2) + ['Mo']*int(n_top/2) + ['S']*int(n_top/2)
        except:
            ### for monolayer ###
            n_bott = self.layer_nsites[0]
            eles = ['Mo']*int(n_bott/2) + ['S']*int(n_bott/2)
        return pmg_struct(self.latt_vec, eles, self.coords, coords_are_cartesian=True)

    def _sites_slice(self):
        layer_norbs_sublatt = np.array(self.layer_norbs_sublatt)*self.spin
        layer_nsites_sublatt = self.layer_nsites_sublatt
        nlayer = len(layer_norbs_sublatt)
        norbs_each_site = [[] for _ in range(nlayer)] 
        for ith_layer in range(nlayer):
            for ith_sublatt in range(len(layer_nsites_sublatt[ith_layer])):
                norbs_each_site[ith_layer] = norbs_each_site[ith_layer] + \
                   [layer_norbs_sublatt[ith_layer][ith_sublatt]]*layer_nsites_sublatt[ith_layer][ith_sublatt]
        norbs_expand = np.concatenate(norbs_each_site)
        return [slice(np.sum(norbs_expand[:i]),np.sum(norbs_expand[:i+1])) for i in range(len(norbs_expand))]

    def add_hopping(self, hop_params_intra=hop_params_intra_MoS2, inter_neigh_pair='S-S',spin=1, dist_cut=8):
        self.spin = spin
        self.site_slice = self._sites_slice()
        intra_hop_mat = IntraHopMat(hop_params_intra, spin)
        onsite_energy = intra_hop_mat.onsite_mat

        pmg_st = self.pymatgen_structure()
        neigh_intra, neigh_inter = get_neighbors_MX2(pmg_st, dist_cut, self.layer_nsites)
        nsite = np.sum(self.layer_nsites)

        ### intralayer hopping energy ###
        def get_angles(neighs, layer):
            neighs = np.array(neighs)
            if layer=='bottom':
                ref_x_aixs = self.ref_x_axis[0]
            elif layer=='top':
                ref_x_aixs = self.ref_x_axis[1]
            ang_ref = np.angle(ref_x_aixs[0]+1j*ref_x_aixs[1])*180/np.pi
            p0 = neighs[:,0]
            p1 = neighs[:,3]
            offset = neighs[:,1:3]
            offset = np.append(offset, [[0]]*len(offset),axis=1)
            r0 = self.coords[p0]
            r1 = self.coords[p1] + frac2cart(offset, self.latt_vec)
            dr = (r1 - r0)[:,0:2]
            dr_ref = rotate_on_vec(-ang_ref, dr)
            #dr_ref = dr
            dr_ref_polar = dr_ref[:,0]+1j*dr_ref[:,1]
            angles = np.array(np.round(np.angle(dr_ref_polar)*180/np.pi), dtype=int)
            return angles
        hop_list = [{} for _ in range(nsite)]
        for layer in neigh_intra:
            try:
                for pair_type in neigh_intra[layer]:
                    angles = get_angles(neigh_intra[layer][pair_type], layer)
                    i = 0
                    for site_pair in neigh_intra[layer][pair_type]:
                        t = intra_hop_mat.get_hop_mat(pair_type, angles[i])
                        hop_list[site_pair[0]][tuple(site_pair[1:])] = t
                        i = i + 1
            except:
                pass
        ### interlayer hopping ###
        try:
            t_inter_hop_mat = InterHop(inter_neigh_pair)
            for i in range(len(neigh_inter[0])):
                r0 = self.coords[neigh_inter[0][i]]
                offset = np.append(np.array(neigh_inter[1][i][0:2]),[0])
                r1 = self.coords[neigh_inter[1][i][2]]+ frac2cart(offset, self.latt_vec)
                #t = t_inter_hop_mat.get_inter_hop_mat(r1-r0, spin, self.ref_x_axis[0], self.ref_x_axis[1])
                ### for interlayer hopping orbital recover to globle axis, namely the bottom layer
                t = t_inter_hop_mat.get_inter_hop_mat(r1-r0, spin, self.ref_x_axis[0], self.ref_x_axis[0])
                hop_list[neigh_inter[0][i]][tuple(neigh_inter[1][i])] = t
        except:
            pass

        self.hoppings = hop_list
        #############################################


        ### onsite energy ###
        n_bott_M = int(self.layer_nsites[0]/2)
        n_bott_X = n_bott_M
        e_on_M = onsite_energy['M']
        e_on_X = onsite_energy['X']
        try:
            n_top_M = int(self.layer_nsites[1]/2)
            n_top_X = n_top_M
            e_onsite = [e_on_M]*n_bott_M + [e_on_X]*n_bott_X + [e_on_M]*n_top_M + [e_on_X]*n_top_X
        except:
            e_onsite = [e_on_M]*n_bott_M + [e_on_X]*n_bott_X
        self.onsite_energy = e_onsite

        norb = 0
        for ith_layer in range(len(self.layer_nsites_sublatt)):
            layeri_nsites_sublatt = np.array(self.layer_nsites_sublatt[ith_layer])
            layeri_norbs_sublatt = np.array(self.layer_norbs_sublatt[ith_layer])
            norb = norb + np.sum(layeri_nsites_sublatt*layeri_norbs_sublatt)
        self.norb = norb

class MX2(_MX2TB_Methods, _PBCMethods):
    def __init__(self, a=3.18, d_XX=3.13, c=12.29):
        """
        MoS2:  a=3.18, d_XX=3.13, c=12.29
        MoSe2: a=3.32, d_XX=3.34, c=12.90
        WS2:   a=3.18, d_XX=3.14, c=12.32
        WSe2:  a=3.32, d_XX=3.35, c=12.96
        """
        self.a = a
        self.h = c/2 - d_XX
        self.d_XX = d_XX
        self._make_structure()

    def _make_structure(self):
        sublatts_frac = np.array([[1/3.,1/3., 0],[2/3.,2/3.,0]])
        self.latt_vec = self.a*np.array([[np.sqrt(3)/2, -1/2., 0.],[np.sqrt(3)/2, 1/2., 0.], [0, 0, 100/self.a]])
        #self.latt_vec = self.a*np.array([[1, 0, 0.],[1/2, np.sqrt(3)/2, 0.], [0, 0, 100/self.a]])
        self.latt_vec_bottom = self.latt_vec[:,0:2]
        self.coords = frac2cart(sublatts_frac, self.latt_vec)
        self.layer_nsites = [len(sublatts_frac)]
        self.layer_nsites_sublatt = [[1,1]]
        self.layer_norbs_sublatt = [[5,6]]
        self.nsite = len(self.coords)
        self.ref_x_axis = [self.latt_vec_bottom[0]]

    def to_bilayer(self, stack='2H'):
        coords = copy.deepcopy(self.coords)
        if stack=='AA':
            coords1 = copy.deepcopy(coords)
            coords1[:,-1] = self.h
            self.coords = np.concatenate([coords, coords1], axis=0)
            self.latt_vec_top = self.latt_vec_bottom
            self.ref_x_axis = [self.latt_vec_bottom[0], self.latt_vec_top[0]]
        elif stack=='2H':
            coords1 = copy.deepcopy(coords[::-1])
            coords1[:,-1] = self.h
            self.coords = np.concatenate([coords, coords1], axis=0)
            self.latt_vec_top = self.latt_vec_bottom
            self.ref_x_axis = [self.latt_vec_bottom[0], -self.latt_vec_top[0]]
        self.nsite = self.nsite + len(coords1)
        self.layer_nsites.append(len(coords1))    
        self.layer_nsites_sublatt = [[1,1],[1,1]]
        self.layer_norbs_sublatt = [[5,6],[5,6]]
        self.latt_vec_top = self.latt_vec_bottom


class MX2TwistedBilayer(_MX2TB_Methods,_PBCMethods):
    def __init__(self, a=3.18, d_XX=3.13, c=12.29, rotate_cent='hole'):
        self.a = a
        self.h = c/2 - d_XX
        self.d_XX = d_XX

        if rotate_cent=='hole':
            self.sublatts_frac = np.array([[1/3.,1/3.],[2/3.,2/3.]])

    def make_structure(self, m, n):
        pmg_sublatts, latt_vec_bottom, latt_vec_top =\
            pmg_sublatts_twisted_bilayer_hexagonal_lattice(m, n, self.a, self.h, self.sublatts_frac)
        self.latt_vec_bottom = latt_vec_bottom[0:2][:,0:2]
        self.latt_vec_top = latt_vec_top[0:2][:,0:2]
        self.ref_x_axis = [self.latt_vec_bottom[0], self.latt_vec_top[0]]
        #return pmg_sublatts
        self.layer_nsites= [0,0]
        for i in range(len(self.sublatts_frac)):
            latt_bott = pmg_sublatts[0][i]
            self.layer_nsites[0] += latt_bott.num_sites
            latt_top = pmg_sublatts[1][i]
            self.layer_nsites[1] += latt_top.num_sites
        self.layer_nsites_sublatt = [[latt.num_sites for latt in pmg_sublatts[0]],[latt.num_sites for latt in pmg_sublatts[1]]]
        self.latt_vec = pmg_sublatts[0][0].lattice.matrix
        coords_bott = np.concatenate([latt.cart_coords for latt in pmg_sublatts[0]], axis=0)
        coords_top = np.concatenate([latt.cart_coords for latt in pmg_sublatts[1]], axis=0)
        self.coords = np.concatenate([coords_bott, coords_top], axis=0)
        self.layer_norbs_sublatt = [[5,6],[5,6]]
        self.nsite = len(self.coords)
 
