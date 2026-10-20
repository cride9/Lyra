"""Measure persistent stack/two-point loops; short intentional repeats remain legal."""
from __future__ import annotations

import math


def _distance(a,b):
    return math.hypot(a.x-b.x,a.y-b.y)


def repetition_report(objects):
    circles=[o for o in objects if o.kind=="circle"]
    best={"stack_ms":0,"bounce_ms":0,"stack_objects":0,"bounce_objects":0,"penalty":0.,"degenerate":False}
    if len(circles)<2:
        return best
    stack_start=bounce_start=0
    for i in range(1,len(circles)):
        current,previous=circles[i],circles[i-1]
        nearby=i>=2 and current.time-circles[i-2].time<=2000
        stack=current.time-previous.time<=1000 and _distance(current,previous)<=14
        bounce=(nearby and current.time-previous.time<=1000 and
                _distance(current,circles[i-2])<=14 and _distance(current,previous)>=64)
        if not stack: stack_start=i
        if not bounce: bounce_start=i
        if stack and i-stack_start+1>=4:
            best["stack_ms"]=max(best["stack_ms"],current.time-circles[stack_start].time)
            best["stack_objects"]=max(best["stack_objects"],i-stack_start+1)
        if bounce and i-bounce_start+1>=6:
            best["bounce_ms"]=max(best["bounce_ms"],current.time-circles[bounce_start].time)
            best["bounce_objects"]=max(best["bounce_objects"],i-bounce_start+1)
    best["penalty"]=max(0,best["stack_ms"]-4000)/4000+max(0,best["bounce_ms"]-6000)/4000
    best["degenerate"]=best["stack_ms"]>=16000 or best["bounce_ms"]>=16000
    return best


def generation_bias(objects,state,indices,tok):
    """Logit penalty after several repeated circles; never bans a single stack/jump."""
    if state not in ("x","y"):
        return None
    circles=[o for o in objects[-24:] if o.kind=="circle"]
    if len(circles)<5:
        return None
    report=repetition_report(circles)
    if report["stack_objects"]<5 and report["bounce_objects"]<6:
        return None
    import torch
    positions=(indices-tok.ranges["XY"][0]-1024)*2
    if report["stack_objects"]>=5:
        coordinate=getattr(circles[-1],state)
        return torch.where((positions-coordinate).abs()<=20,-min(7.,2.+report["stack_objects"]*.45),0.)
    coordinate=getattr(circles[-2],state)
    return torch.where((positions-coordinate).abs()<=20,-min(7.,2.+report["bounce_objects"]*.4),0.)
