"""Offline pulse evidence, octave candidates and robust piecewise tempo fits."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, fftconvolve

from . import audio, mapio


class TimingUncertain(ValueError):
    def __init__(self,report):
        self.report=report
        super().__init__(report.get("warning","Automatic timing is uncertain; provide BPM/offset or a timing map"))


def estimate(scores, frame_ms=audio.FRAME_MS, bpm=None, offset=None, strict=True):
    if bpm is not None:
        if not 20<=bpm<=400: raise ValueError("BPM must be between 20 and 400")
        return [mapio.TimingPoint(float(offset or 0),60000/bpm)],{"manual":True,"confidence":None,"bpm":bpm}
    scores=np.asarray(scores,dtype=np.float64)
    if scores.ndim!=2 or scores.shape[1]!=2 or not np.isfinite(scores).all(): raise ValueError("Invalid timing activations")
    n=len(scores); segment=max(1,round(32000/frame_ms)); fits=[]; coverage=0
    for start in range(0,n,segment):
        values=scores[start:min(n,start+segment),0]
        peaks,_=find_peaks(values,height=.15,prominence=.03,distance=max(1,round(110/frame_ms)))
        if len(peaks)<5: continue
        signal=np.maximum(values-np.percentile(values,30),0)
        ac=fftconvolve(signal,signal[::-1],mode="full")[len(signal)-1:]
        low=max(1,round(150/frame_ms)); high=min(len(ac)-1,round(3000/frame_ms))
        if high<=low: continue
        ac=ac/np.maximum(np.arange(len(signal),0,-1),1)
        maxima,_=find_peaks(ac[low:high])
        lags=(maxima+low)[np.argsort(ac[maxima+low])[-12:]]
        periods=set()
        for lag in lags:
            for multiple in (.5,1,2):
                period=float(lag*multiple*frame_ms)
                if 150<=period<=3000: periods.add(period)
        best=None; peak_times=peaks*frame_ms
        for period in sorted(periods):
            for seed in peaks[np.argsort(values[peaks])[-12:]]:
                origin=seed*frame_ms
                grid=np.arange(origin-np.ceil(origin/period)*period,len(values)*frame_ms,period)
                grid=grid[grid>=0]
                if len(grid)<5: continue
                distances=np.abs(grid[:,None]-peak_times[None,:]); nearest=distances.argmin(1)
                good=distances.min(1)<max(30,period*.08)
                if good.sum()<5: continue
                indices=np.arange(len(grid))[good]; hits=peak_times[nearest[good]]
                fitted,intercept=np.polyfit(indices,hits,1)
                if fitted<=0 or abs(fitted-period)>period*.08: continue
                error=np.abs(hits-(intercept+indices*fitted))
                inliers=error<35
                if inliers.sum()<5: continue
                fitted,intercept=np.polyfit(indices[inliers],hits[inliers],1)
                grid=intercept+np.arange(len(grid))*fitted
                strength=np.interp(grid,np.arange(len(values))*frame_ms,values,left=0,right=0).mean()
                matched=len(set(nearest[good][inliers]))/len(peaks)
                quality=.6*strength+.25*float(good.mean())+.15*matched
                if best is None or quality>best[0]: best=(quality,fitted,intercept,float(np.median(error[inliers])),grid)
        if best is None: continue
        quality,period,phase,error,grid=best
        # Use actual downbeat evidence; do not claim a measured meter if absent.
        down=np.interp(grid,np.arange(len(values))*frame_ms,scores[start:start+len(values),1])
        meter_scores=[(float(down[k::4].mean()) if len(down[k::4]) else 0,k) for k in range(4)]
        _,bar_phase=max(meter_scores)
        phase+=bar_phase*period
        while phase>0: phase-=4*period
        fits.append({"time":start*frame_ms+phase,"period":period,"confidence":float(np.clip(quality,0,1)),"error_ms":error,"segment_start":start*frame_ms})
        coverage+=len(values)
    points=[]
    for f in fits:
        if points:
            previous=points[-1]
            relative=(f["time"]-previous.time)/previous.beat_length
            phase_error=abs(relative-round(relative))*previous.beat_length
            if abs(f["period"]/previous.beat_length-1)<.01 and phase_error<25: continue
        at=f["time"]
        if points and at<=points[-1].time: at=f["segment_start"]
        points.append(mapio.TimingPoint(float(at),float(f["period"])))
    confidence=float(np.mean([f["confidence"] for f in fits])) if fits else 0
    report={"confidence":confidence,"coverage":coverage/max(1,n),"segments":fits,"uncertain":not points or confidence<.55 or coverage/max(1,n)<.5}
    if report["uncertain"]:
        report["warning"]="Insufficient reliable beat evidence. Preview the metronome and correct BPM/offset or supply a timing map. No default BPM was substituted."
        if strict: raise TimingUncertain(report)
    if points and offset is not None:
        shift=offset-points[0].time
        for point in points: point.time+=shift
    return points,report


def metronome(audio_path, timing, destination):
    import soundfile as sf
    pcm=audio.decode(audio_path)
    click=np.sin(2*np.pi*1600*np.arange(round(.025*audio.SR))/audio.SR)*np.exp(-np.arange(round(.025*audio.SR))/(audio.SR*.006))*.35
    for i,p in enumerate(timing):
        end=timing[i+1].time if i+1<len(timing) else len(pcm)*1000/audio.SR
        for at in np.arange(p.time,end,p.beat_length):
            a=round(at*audio.SR/1000); left=max(0,a); right=min(len(pcm),a+len(click))
            if right>left: pcm[left:right]+=click[left-a:right-a]
    sf.write(destination,np.clip(pcm,-1,1),audio.SR)
    return str(destination)
