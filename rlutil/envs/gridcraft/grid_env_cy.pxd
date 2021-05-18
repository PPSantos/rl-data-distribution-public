from rlutil.envs.tabular_cy cimport tabular_env
from rlutil.envs.gridcraft cimport grid_spec_cy
from libcpp.map cimport map

cdef class GridEnv(tabular_env.TabularEnv):
    cdef grid_spec_cy.GridSpec gs
    cdef double phi
    cdef map[int, double] _deterministic_transition_map
    cdef map[int, double] _transitions_cy(self, int state, int action)
    cpdef set_phi(self, double phi)

cdef class DistanceRewardGridEnv(GridEnv):
    cdef double start_dist
    cdef int rew_x
    cdef int rew_y