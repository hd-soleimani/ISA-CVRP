#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:49:47 2020

@author: yas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.cluster import DBSCAN
import xml.etree.ElementTree as ET
import re
import glob, os
from helpers.cvrp_rkm16_io import calculate_D
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import xml.dom.minidom
from xml.etree.ElementTree import Element,SubElement,Comment,tostring,dump
from xml.dom.minidom import parseString
from xml.sax.saxutils import unescape
import xlwt
import math  
from sklearn import preprocessing
from scipy.spatial.distance import pdist, cdist, squareform
from pandas import ExcelWriter    


from math import radians, cos, sin, asin, sqrt
from operator import itemgetter
from itertools import groupby


style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on',
num_format_str='#,##0.00')
style1 = xlwt.easyxf(num_format_str='D-MMM-YY')
wb = xlwt.Workbook()
ws = wb.add_sheet('A Test Sheet')
ws.write(0, 0, 'Name', style0)
ws.write(0, 1, "Number of Var", style0)
ws.write(0, 2, "Obj. Fun.", style0)
ws.write(0, 3, 'MIP Gap', style0)
ws.write(0, 4, 'Time (s)', style0)




iteration=0

for infile in sorted(glob.glob(os.path.join( '*.xml'))):
#for infile in sorted(glob.glob(os.path.join( 'Outlier_Double_Distance_CVRP238.xml'))):
    print "Current File Being Processed is: " + infile
    tree = ET.parse(infile)
    root = tree.getroot()
    iteration+=1

    
#    tree = ET.parse(file_name)
#    root = tree.getroot()

    edge_weight_type='EUC_2D'
#Dataset
    for dataset in root.iter('dataset'):
        print(dataset.text)
        
#Name
    for name in root.iter('name'):
        print(name.text)   

#Capacity
    for capacity in root.iter('capacity'):
        C=float(capacity.text)
        print(C);
   
#Finding the number of nodes
    l=0;
    for quantity in root.iter('quantity'):
        l=l+1
#    print(j)
    N=l
    
#Demands
#    demands =[{} for x in range( N+1 ) ];
    demands=[None] * (N+1)
    i=1;
    for quantity in root.iter('quantity'):
        demands[i]=float(quantity.text)
        i=i+1
    demands[0]=0
    demandssort=np.sort(demands)
    
#    print(demands)
    
#Depot position and nodes positions (depot in posx[0] and posy[0])
    #posx
#    posx =[{} for x in range( N+1 ) ]
#    posy =[{} for x in range( N+1) ]
    posx =[None] * (N+1)
    posy =[None] * (N+1)
    i=0
    for cx in root.iter('cx'):
        posx[i]=float(cx.text)
        i=i+1
#    print(posx)

    #posy
    i=0
    for cy in root.iter('cy'):
        posy[i]=float(cy.text)
        i=i+1
#    print(posy)
    depot=(posx[0],posy[0])
#Coordinations
    points=[];
    for i in range( N+1 ):
        one = posx[i]
        two = posy[i]
        points.append((one, two))
#    points=[[posx],[posy]]
#    print(points)
    dd_points=points    

#Distance Matrix
    D=calculate_D(points, None, edge_weight_type )
#    print(D)
#
    namedtuple('AdditionalConstraints',
'vehicle_count_constraint maximum_route_cost_constraint service_time_at_customer')
    ProblemDefinition = namedtuple('ProblemDefinition',
['size', 'coordinate_points', 'display_coordinate_points',
 'customer_demands', 'distance_matrix', 'capacity_constraint', 'edge_weight_type'])        
        
#        return ProblemDefinition(N, points, dd_points, demands, D, C, edge_weight_type)
    
    
    #Cluster Finding
    #removing depot
    points_cust = np.delete(points,0,axis=0)
#    print(points_cust)
    dataset=points_cust
 
    #Centroid function
    coordinatecenter=[]
    coordinatecenternew=[]


    def findCenter (xlist,ylist,coordinates):
        xcenter=np.sum(xlist)/len(xlist)
        ycenter=np.sum(ylist)/len(ylist)
        coordinates.append(xcenter)
        coordinates.append(ycenter)
        return coordinates

    findCenter (posx, posy, coordinatecenter)
    np.array(coordinatecenter)
#    print(coordinatecenter)





    posxnew=posx
    posynew=posy
    ptsmat=np.insert(points_cust,0,depot, axis=0)
#    ptsmat=np.insert(ptsmat,2,demands, axis=1)



    n=N #number of customers


#import the dataset, the first is the depot, the other is the customers
#dataset=pd.read_csv('CVRP674.csv')

    X=posx
    Y=posy



    N=[i for i in range (1,n+1)]
    V=[0]+N
    A=[(i,j) for i in V for j in V if i!=j]

    c= {(i,j): distance.euclidean([X[i],Y[i]], [X[j],Y[j]]) for i,j in A}
    Q=C

    #demands=pd.read_csv('demands.csv')

    q=demands
    #q= [0,10,30,20,20,30,10,40,20,20,20,10,20,30,40,20,10,10,20,20,20,20,40,10,10,40,30,10,20]

    from gurobipy import Model, GRB, quicksum

    mdl=Model('CVRP')

    x=mdl.addVars(A, vtype=GRB.BINARY)
    u=mdl.addVars(N, vtype=GRB.CONTINUOUS)

    mdl.modelSense=GRB.MINIMIZE
    mdl.setObjective(quicksum(x[i,j]*c[i,j] for i,j in A))

    mdl.addConstrs(quicksum(x[i,j] for j in V if j!=i)==1 for i in N );
    mdl.addConstrs(quicksum(x[i,j] for i in V if i!=j)==1 for j in N );

    #subtour elimination
    mdl.addConstrs((x[i,j]==1) >> (u[i]+q[j]==u[j]) for i,j in A if i!=0 and j!=0);

    mdl.addConstrs(u[i]>=q[i] for i in N);
    mdl.addConstrs(u[i]<=Q for i in N);



#termination rules:
    #https://www.gurobi.com/documentation/9.0/refman/mip_models.html
    
    #the number of discovered feasible integer solutions exceeds the specified value
    mdl.Params.SolutionLimit=3
    
    #mdl.Params.MIPGap=0.2
    #mdl.Params.TimeLimit=10
    mdl.optimize()

    activate_arcs=[a for a in A if x[a].x>0.99]
    
    #Attributes:
    #https://www.gurobi.com/documentation/9.0/refman/attributes.html



#    for i,j in activate_arcs:
#        plt.plot([X[i],X[j]], [Y[i],Y[j]],c='g',zorder=0)
#
#    plt.plot(X[0],Y[0],c='r',marker='s')
#    plt.scatter(X[1:],Y[1:],c='b')
#    plt.title('CVRP674:  Distance: Half, Demand: Hard  ')
#    #plt.title('CVRP667: ORIGINAL  ')

    objective=mdl.objVal
    gap=mdl.MIPGap
    variables=mdl.NumVars
    time=mdl.Runtime
    ws.write(iteration, 0, str(infile), style0)
    ws.write(iteration, 1, variables, style0)
    ws.write(iteration, 2, objective, style0)
    ws.write(iteration, 3, gap, style0)
    ws.write(iteration, 4, time, style0)


    wb.save('Results.xls')

