"""Response-blind input matching for exact, local and joint tests."""
import numpy as np
from . import PARAMETERS as P


class Matches:
    def __init__(self,pool,history):
        self.pool,self.history = pool,history
        self.cache = {}
        self.distance = []
        for observation in history:
            point = pool.features[pool.ix[observation['candidate_id']]]
            columns = []
            for axis,kind in enumerate(pool.kinds):
                if kind == 'continuous':
                    values = np.asarray([f[axis] for f in pool.features],float)
                    span = max(float(np.ptp(values)),1e-12)
                    columns.append(np.abs(values-point[axis])/span)
                else:
                    columns.append(np.asarray([f[axis]!=point[axis] for f in pool.features],float))
            self.distance.append(np.column_stack(columns))
        if not self.distance:
            raise ValueError('Matching requires existing measurements')
        self.distance = np.asarray(self.distance)

    def for_axes(self,axes):
        axes = tuple(sorted(set(axes)))
        if not axes or not set(axes) <= set(range(len(self.pool.axes))):
            raise ValueError('A nonempty valid focal axis set is required')
        if axes in self.cache:
            return self.cache[axes]
        n = len(self.pool.ids); other = [i for i in range(len(self.pool.axes)) if i not in axes]
        best = np.full(n,np.inf); focal_best = best.copy(); anchor = np.full(n,-1,int)
        for i,distance in enumerate(self.distance):
            nuisance = np.sqrt((distance[:,other]**2).mean(axis=1)) if other else np.zeros(n)
            focal = np.sqrt((distance[:,axes]**2).sum(axis=1))
            # Repeats do not qualify as experimental contrasts.
            valid = focal > P['continuous_tolerance']
            take = valid & ((nuisance < best) | ((nuisance == best)&(focal <= focal_best)))
            best[take],focal_best[take],anchor[take] = nuisance[take],focal[take],i
        quality = np.exp(-.5*(best/P['matching_bandwidth'])**2)
        self.cache[axes] = dict(quality=quality,nuisance=best,anchor=anchor,focal_distance=focal_best)
        return self.cache[axes]

    def describe(self,axes,index):
        values = self.for_axes(axes); i = int(values['anchor'][index])
        if i < 0:
            return dict(anchor_id=None,focus_axes=[self.pool.axes[a] for a in axes],
                kind='no_changed_focal_coordinate',quality=0.,nuisance_RMS=None)
        nuisance = float(values['nuisance'][index])
        kind = ('exact_focal_contrast' if nuisance <= P['continuous_tolerance'] else
            'local_match' if nuisance <= P['local_match_RMS'] else 'joint_condition_test')
        return dict(anchor_id=self.history[i]['candidate_id'],focus_axes=[self.pool.axes[a] for a in axes],
            kind=kind,quality=float(values['quality'][index]),nuisance_RMS=nuisance,
            focal_distance=float(values['focal_distance'][index]))
