"""Read-only official floor triangle containment for the C2 geometry pilot.

Uses the same interval-clipping rule as src/bvi/pose_goal.py FloorMesh.
It checks a point or straight path against static floor triangles only.
"""
import hashlib
import math
from pathlib import Path


class FloorTriangles:
    def __init__(self, path):
        self.path=Path(path)
        vertices=[];faces=[]
        for line in self.path.read_text().splitlines():
            parts=line.split()
            if not parts:continue
            if parts[0]=='v':vertices.append(tuple(map(float,parts[1:3])))
            if parts[0]=='f':faces.append([int(x.split('/')[0])-1 for x in parts[1:]])
        if not vertices or not faces or any(len(f)!=3 or any(i<0 or i>=len(vertices) for i in f) for f in faces):
            raise ValueError('Invalid official triangular floor mesh')
        self.triangles=[tuple(vertices[i] for i in f) for f in faces]
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()

    def contains_segment(self,start,end):
        def cross(a,b):return a[0]*b[1]-a[1]*b[0]
        d=(end[0]-start[0],end[1]-start[1]);intervals=[]
        for triangle in self.triangles:
            a,b,c=triangle
            orient=cross((b[0]-a[0],b[1]-a[1]),(c[0]-a[0],c[1]-a[1]))
            if abs(orient)<1e-12:continue
            sign=1 if orient>0 else -1;lo=0.;hi=1.
            for a,b in zip(triangle,(*triangle[1:],triangle[0])):
                edge=(b[0]-a[0],b[1]-a[1])
                v=sign*cross(edge,(start[0]-a[0],start[1]-a[1]))
                rate=sign*cross(edge,d)
                if abs(rate)<1e-12:
                    if v< -1e-9:hi=-1.;break
                elif rate>0:lo=max(lo,(-1e-9-v)/rate)
                else:hi=min(hi,(-1e-9-v)/rate)
            if lo<=hi:intervals.append((lo,hi))
        covered=0.
        for lo,hi in sorted(intervals):
            if lo>covered+1e-7:return False
            covered=max(covered,hi)
            if covered>=1.-1e-7:return True
        return False
