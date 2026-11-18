"""Adjacent-support plane hypotheses, selected before observing later coverage.

Adapted from the current geometry_pilot_v2 implementation; no dataset paths.
"""
import cv2
import numpy as np

CONFIG = {
    "patch_size_px":20, "patch_stride_px":16, "ring_radius_px":90,
    "plane_distance_m":.008, "normal_angle_deg":12., "patch_residual_m":.0025,
    "min_connected_support_pixels":300, "min_object_above_fraction":.9,
}

def normals(X,valid):
    dx=np.roll(X,-4,axis=1)-np.roll(X,4,axis=1)
    dy=np.roll(X,-4,axis=0)-np.roll(X,4,axis=0)
    N=np.cross(dx,dy);length=np.linalg.norm(N,axis=2)
    N/=np.maximum(length[:,:,None],1e-10)
    N[np.sum(N*X,axis=2)>0]*=-1
    good=valid.copy() & (length>1e-6)
    for axis in [0,1]:
        for delta in [-4,4]:good &= np.roll(valid,delta,axis=axis)
    good[:4]=False;good[-4:]=False;good[:,:4]=False;good[:,-4:]=False
    good&=(np.linalg.norm(dx,axis=2)<.15)&(np.linalg.norm(dy,axis=2)<.15)
    return N,good


def fit_svd(P):
    center=P.mean(0);_,_,vh=np.linalg.svd(P-center,full_matrices=False)
    n=vh[-1]
    if n@center>0:n=-n
    return n,float(-n@center)


def connected_support(mask,near=None):
    joined=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(joined,8)
    if count<2:return np.zeros_like(mask)
    best=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    if near is not None:
        # Prefer actual adjacent support over a larger disconnected remote patch.
        eligible=[k for k in range(1,count) if np.count_nonzero((labels==k)&mask)>=CONFIG['min_connected_support_pixels']
                  and np.count_nonzero((labels==k)&mask&near)>=50]
        if eligible:best=max(eligible,key=lambda k:np.count_nonzero((labels==k)&mask))
    return (labels==best)&mask


def plane_hypotheses(X,valid,object_mask,dynamic,additional_seeds=None,support_normal_hint=None,prefer_adjacent_component=False):
    """Generate on start image only; rank by measured connected support, not coverage."""
    m=object_mask>0;diameter=2*CONFIG['ring_radius_px']+1
    outer=cv2.dilate(m.astype('uint8'),np.ones((diameter,diameter),np.uint8))>0
    inner=cv2.dilate(m.astype('uint8'),np.ones((17,17),np.uint8))>0
    N,ngood=normals(X,valid)
    ring=outer&~inner&(dynamic==0)&ngood
    cos=np.cos(np.deg2rad(CONFIG['normal_angle_deg']))
    seeds=[];h,w=ring.shape
    size=CONFIG['patch_size_px'];stride=CONFIG['patch_stride_px']
    for y in range(0,h-size,stride):
        for x in range(0,w-size,stride):
            mask=ring[y:y+size,x:x+size]
            if mask.sum()<.7*size*size:continue
            P=X[y:y+size,x:x+size][mask];n,d=fit_svd(P)
            residual=np.abs(P@n+d)
            if np.quantile(residual,.9)>CONFIG['patch_residual_m']:continue
            if np.mean(N[y:y+size,x:x+size][mask]@n>cos)<.8:continue
            # Merge near-identical local hypotheses before expensive image evaluation.
            center=P.mean(0)
            if any(n@nn>np.cos(np.deg2rad(4)) and abs(center@nn+dd)<.005 for nn,dd in seeds):continue
            seeds.append((n,d))
    initial_seed_count=len(seeds)
    for seed in additional_seeds or []:
        n=np.asarray(seed['normal'],dtype=float);d=float(seed['offset'])
        if n.shape!=(3,) or not np.isfinite(n).all() or not np.isfinite(d) or not np.isclose(np.linalg.norm(n),1.,atol=1e-5):
            raise ValueError('Invalid auxiliary support seed')
        if not any(n@nn>np.cos(np.deg2rad(4)) and abs(d-dd)<.005 for nn,dd in seeds):seeds.append((n,d))
    yy,xx=np.where(ring);P=X[yy,xx];NN=N[yy,xx]
    candidates=[];obj=X[m&valid];rejections=[]
    near=cv2.dilate(m.astype('uint8'),np.ones((49,49),np.uint8))>0
    for n,d in seeds:
        support=np.zeros_like(ring)
        for _ in range(2):
            support[:]=False
            yes=(np.abs(P@n+d)<CONFIG['plane_distance_m'])&(NN@n>cos)
            support[yy[yes],xx[yes]]=True;support=connected_support(support,near if prefer_adjacent_component else None)
            if support.sum()<CONFIG['min_connected_support_pixels']:break
            n,d=fit_svd(X[support])
        if support.sum()<CONFIG['min_connected_support_pixels']:
            rejections.append({'reason':'too_few_connected_support_pixels','support_pixels':int(support.sum())})
            continue
        if support_normal_hint is not None:
            hint=np.asarray(support_normal_hint['normal'],dtype=float)
            alignment=float(n@hint)
            if alignment<np.cos(np.deg2rad(support_normal_hint['max_angle_deg'])):
                rejections.append({'reason':'incompatible_with_observed_support_orientation','support_pixels':int(support.sum()),'normal_alignment':alignment})
                continue
        above=float(np.mean(obj@n+d>-CONFIG['plane_distance_m'])) if len(obj) else 0
        support_gap=float(np.quantile(obj@n+d,.05)) if len(obj) else float('inf')
        if support_gap>.08:
            rejections.append({'reason':'plane_not_in_contact_with_initial_object',
                               'object_support_gap_p05_m':support_gap})
            continue
        # A supporting plane should lie beneath the reference object, not cut it.
        if above<CONFIG['min_object_above_fraction']:
            rejections.append({'reason':'plane_cuts_reference_object','support_pixels':int(support.sum()),'object_above_fraction':above})
            continue
        contact=int((support&near).sum())
        if contact<50:
            rejections.append({'reason':'support_not_adjacent_to_reference','support_pixels':int(support.sum()),'near_object_pixels':contact,'object_above_fraction':above})
            continue
        if any(n@c['normal']>np.cos(np.deg2rad(4)) and abs(d-c['offset'])<.008 for c in candidates):continue
        residual=np.abs(X[support]@n+d)
        candidates.append({'normal':n,'offset':d,'support_mask':support,
                           'support_pixels':int(support.sum()),'near_object_pixels':contact,
                           'object_above_fraction':above,'median_residual_m':float(np.median(residual))})
    candidates.sort(key=lambda c:c['support_pixels'],reverse=True)
    return candidates[:5],{'patch_hypotheses':len(seeds),'initial_patch_hypotheses':initial_seed_count,'additional_seed_count':len(seeds)-initial_seed_count,'candidate_count':len(candidates),'ring_pixels':int(ring.sum()),'rejected_candidates':rejections}
