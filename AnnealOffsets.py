#=================================================IMPORTS=========================================================#
print("Started Program...")
import dwave
import dimod
import math
import json
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from dwave.system.samplers import DWaveSampler
from dwave.embedding.pegasus import find_clique_embedding
import pandas as pd 
#import numpy as numpy #it's been a long day 
import numpy as np
import time
import os

token = "put your token here :)"
computer = "Advantage_system4.1"

#=================================================SETUP=========================================================#

sampler = DWaveSampler(token=token, solver=computer)
hardware_graph = dnx.pegasus_graph(16, node_list=sampler.properties['qubits'], edge_list=sampler.edgelist)
A_matrix = sampler.adjacency #This is the adjacency matrix

#================================================JSON READER=====================================================#

runMode = "QPU"

#2V40T
#inputFile = [[0.619949839686173, [[0.6201755054922214, 0.0004166666666666667], [0.6192006233972701, 0.0005833333333333333], [0.6207376504040393, 0.0004166666666666667], [0.6190406640308416, 0.0009166666666666668], [0.6202956081170473, 0.0004166666666666667], [0.6204226895306945, 0.0009166666666666668], [0.619552084928523, 0.0003333333333333334], [0.620112858971073, 0.0004166666666666667], [0.6196292976372996, 0.0013833333333333334], [0.621064412590791, 0.00205], [0.6209727807884767, 0.0013833333333333334], [0.6202057685272748, 0.0009166666666666668], [0.6205460316941176, 0.0005833333333333333], [0.6200698907957599, 0.0004166666666666667], [0.6184321606871547, 0.0013833333333333334], [0.6196795986094037, 0.0004166666666666667], [0.6198699251659776, 0.0013833333333333334], [0.6198822289009036, 0.0005833333333333333], [0.6195363137659851, 0.0004166666666666667], [0.620667676692613, 0.0004166666666666667]]]                        , [0.7644644052662664, [[0.7648474096270056, 0.0005833333333333333], [0.764629660488086, 0.0003333333333333334], [0.7634440363682118, 0.0004166666666666667], [0.7654512098472598, 0.0013833333333333334], [0.764528127739269, 0.0004166666666666667], [0.7642966788475329, 0.0013833333333333334], [0.764755000398421, 0.0005833333333333333], [0.7652707083245642, 0.0005833333333333333], [0.7644544000797732, 0.0004166666666666667], [0.7640091833055777, 0.0013833333333333334], [0.7648443061117294, 0.0004166666666666667], [0.7654803235039588, 0.0005833333333333333], [0.7644352157389597, 0.0003333333333333334], [0.7626683541781973, 0.0009166666666666668], [0.7639046361035419, 0.0009166666666666668], [0.7647682664914893, 0.0009166666666666668], [0.7633141963245973, 0.0009166666666666668], [0.7651315503772118, 0.0004166666666666667], [0.7654841850315482, 0.0009166666666666668], [0.7653116213608953, 0.0009166666666666668]]]]
#3V6T
#inputFile = [[0.5230456769168634, [[0.5240341355642414, 0.0009166666666666668], [0.5235322589723301, 0.0004166666666666667]]], [0.4017284489848494, [[0.4023188409447569, 0.0004166666666666667], [0.4015596160807172, 0.0009166666666666668]]], [0.46472244197733575, [[0.4650397642828156, 0.0004166666666666667], [0.46458608339970275, 0.0013833333333333334]]]]
#2V4T
#inputFile = [[0.3576734259547776, [[0.3578421714920063, 0.0003333333333333334], [0.35763199025343234, 0.0003333333333333334]]], [0.43751589332926805, [[0.4378207965666029, 0.0005833333333333333], [0.4375708301205863, 0.0004166666666666667]]]]
#2V50T
#inputFile = [[0.40372712245907655, [[0.40313230894383995, 0.0005833333333333333], [0.40128689560282804, 0.0013833333333333334], [0.40411536912117757, 0.0003333333333333334], [0.40229746198479316, 0.0013833333333333334], [0.4026530582972256, 0.0005833333333333333], [0.40496019263698085, 0.00205], [0.4035448749007843, 0.0005833333333333333], [0.40330605065684577, 0.0009166666666666668], [0.40395058872927236, 0.0009166666666666668], [0.4046022135805178, 0.0005833333333333333], [0.40292238395410696, 0.0009166666666666668], [0.4023013958592916, 0.0009166666666666668], [0.40286435832905343, 0.00205], [0.4046331997053262, 0.0004166666666666667], [0.4037195224233967, 0.0005833333333333333], [0.40289850548321177, 0.0003333333333333334], [0.40256632631597933, 0.00205], [0.40284779126459225, 0.0009166666666666668], [0.4030053912344479, 0.0005833333333333333], [0.4073209909011575, 0.00205], [0.4047261154188145, 0.00205], [0.4039366745709197, 0.0004166666666666667], [0.40355421844393713, 0.0003333333333333334], [0.4041819489326438, 0.0005833333333333333], [0.40248173744565413, 0.0013833333333333334]]], [0.46567486036165345, [[0.4641909456987078, 0.00205], [0.4656847208660851, 0.0004166666666666667], [0.4651644822671265, 0.0005833333333333333], [0.46520789913516536, 0.0003333333333333334], [0.4654909585992463, 0.0004166666666666667], [0.46478660256079696, 0.0009166666666666668], [0.46403649879472936, 0.0013833333333333334], [0.46554608568856276, 0.0003333333333333334], [0.4646012247510719, 0.0004166666666666667], [0.4658324856307101, 0.0003333333333333334], [0.4667948738926254, 0.00205], [0.46429959512639873, 0.00205], [0.465264575691102, 0.0004166666666666667], [0.46611862512493846, 0.0003333333333333334], [0.4653225360875454, 0.0013833333333333334], [0.4661083934494567, 0.0003333333333333334], [0.46444244799491674, 0.0005833333333333333], [0.46617339671279684, 0.0003333333333333334], [0.46583324598545045, 0.0005833333333333333], [0.4653467112466013, 0.0009166666666666668], [0.46659589148806285, 0.00205], [0.4657853869436152, 0.0003333333333333334], [0.4654675477451264, 0.0005833333333333333], [0.46531140580563884, 0.0003333333333333334], [0.46541815969045824, 0.0004166666666666667]]]]
#2V10T
#inputFile = [[0.26459440140957663, [[0.2647385486134077, 0.0003333333333333334], [0.2654162874927705, 0.0005833333333333333], [0.2644212460179555, 0.0004166666666666667], [0.26479940986166234, 0.0005833333333333333], [0.26483538260019035, 0.0003333333333333334]]], [0.6371146815924992, [[0.6370521845916481, 0.0005833333333333333], [0.6371294827443515, 0.0005833333333333333], [0.639912825546594, 0.00205], [0.6365240549681587, 0.0005833333333333333], [0.6375382756189499, 0.0005833333333333333]]]]
#2V16T
#inputFile = [[0.3477026272590622, [[0.34670542276621336, 0.0013833333333333334], [0.3479626564874828, 0.0005833333333333333], [0.34719922931666236, 0.0013833333333333334], [0.34582910050787274, 0.0013833333333333334], [0.3484048734552079, 0.0005833333333333333], [0.347763739911817, 0.0005833333333333333], [0.3465146075655664, 0.0009166666666666668], [0.3478775642040726, 0.0013833333333333334]]], [0.5096619032548017, [[0.5093428196876093, 0.0005833333333333333], [0.5095549411829058, 0.0009166666666666668], [0.5102061296299512, 0.0005833333333333333], [0.5087631763603003, 0.0003333333333333334], [0.5086694400220225, 0.0005833333333333333], [0.5098174300893572, 0.0004166666666666667], [0.5094418005132764, 0.0009166666666666668], [0.5093916323048657, 0.0004166666666666667]]]]
#2V20T
#inputFile = [[0.4078385649403785, [[0.4081188272891417, 0.0004166666666666667], [0.4071640022677991, 0.00205], [0.40937063788351874, 0.00205], [0.4055156250132842, 0.0013833333333333334], [0.4081755378876359, 0.0003333333333333334], [0.40748571089521457, 0.0005833333333333333], [0.4076608982535621, 0.0003333333333333334], [0.4076349964133994, 0.0009166666666666668], [0.4082076368274103, 0.0009166666666666668], [0.40782454841127336, 0.0003333333333333334]]], [0.7471953935509581, [[0.748029799603348, 0.0004166666666666667], [0.7464826797517217, 0.0009166666666666668], [0.7473419823192881, 0.0005833333333333333], [0.749613130971712, 0.0013833333333333334], [0.7463983724219616, 0.0009166666666666668], [0.7455691011411218, 0.0013833333333333334], [0.7473114640837099, 0.0005833333333333333], [0.7476735588979205, 0.0013833333333333334], [0.74694666403117, 0.0005833333333333333], [0.7481973954632878, 0.0009166666666666668]]]]
#2V28T
#inputFile = [[0.31571817267074337, [[0.3156134039706826, 0.0003333333333333334], [0.31455038984443007, 0.00205], [0.3157642770348872, 0.00205], [0.31612599297646893, 0.0003333333333333334], [0.3151971311639035, 0.0004166666666666667], [0.3157155546348742, 0.0004166666666666667], [0.3140892584306134, 0.0013833333333333334], [0.3174205985673443, 0.00205], [0.3161015887926657, 0.0009166666666666668], [0.31716675070661854, 0.0009166666666666668], [0.3155865349881447, 0.0009166666666666668], [0.31493158820318123, 0.0005833333333333333], [0.31719940317050427, 0.0009166666666666668], [0.3170537738756036, 0.0013833333333333334]]], [0.661181258203091, [[0.6596810060112318, 0.0013833333333333334], [0.6614027663673783, 0.0004166666666666667], [0.6612599285720642, 0.0003333333333333334], [0.6616816681316353, 0.0005833333333333333], [0.6614533910848687, 0.0003333333333333334], [0.6613573805812722, 0.0013833333333333334], [0.6616798816899587, 0.0013833333333333334], [0.6594706304599387, 0.00205], [0.6609596580784179, 0.0005833333333333333], [0.661268871267803, 0.0005833333333333333], [0.6611259731926012, 0.0013833333333333334], [0.661821177686621, 0.0003333333333333334], [0.6612670836834323, 0.0003333333333333334], [0.6612472160140108, 0.0004166666666666667]]]]
#2V20T
#inputFile = [[0.4078385649403785, [[0.4081188272891417, 0.0004166666666666667], [0.4071640022677991, 0.00205], [0.40937063788351874, 0.00205], [0.4055156250132842, 0.0013833333333333334], [0.4081755378876359, 0.0003333333333333334], [0.40748571089521457, 0.0005833333333333333], [0.4076608982535621, 0.0003333333333333334], [0.4076349964133994, 0.0009166666666666668], [0.4082076368274103, 0.0009166666666666668], [0.40782454841127336, 0.0003333333333333334]]], [0.7471953935509581, [[0.748029799603348, 0.0004166666666666667], [0.7464826797517217, 0.0009166666666666668], [0.7473419823192881, 0.0005833333333333333], [0.749613130971712, 0.0013833333333333334], [0.7463983724219616, 0.0009166666666666668], [0.7455691011411218, 0.0013833333333333334], [0.7473114640837099, 0.0005833333333333333], [0.7476735588979205, 0.0013833333333333334], [0.74694666403117, 0.0005833333333333333], [0.7481973954632878, 0.0009166666666666668]]]]
#2V10T Low Dunn Index
inputFile = [[0.32296819860517706, [[0.32197665057080954, 0.0005833333333333333], [0.32336365050202115, 0.0005833333333333333], [0.32262900734239097, 0.0004166666666666667], [0.3232718569516563, 0.0003333333333333334], [0.322540696715387, 0.0009166666666666668]]], [0.3087398800132073, [[0.30930085225856446, 0.0005833333333333333], [0.30920146256356457, 0.0003333333333333334], [0.3049116217517205, 0.00205], [0.3086836712166612, 0.0003333333333333334], [0.30817878857685843, 0.0009166666666666668]]]]

trackList = [] #This is just a big list of the unclustered data this will be the input to the function to actually cluster them
errorList = [] #List of all error in the same order as the previous list.
location = []

for set in inputFile:
    location.append(set[0])
    for track in set[1]:
        trackList.append(track[0])
        errorList.append(track[1])
numOfClusters = len(location) #This is the variable for how many clusters will be calculated. This can either be an input or decided from
probTitle = str(numOfClusters) + "V" + str(len(trackList)) + "T"
embedding_filename = probTitle + "_" + computer
timeEstimate = (1.5992 * (len(trackList) * numOfClusters)**2 + 12.282 * (len(trackList) * numOfClusters) + 13.817) / 60

configText = probTitle + "\n" + computer

#===============================================METHODS============================================================#

Q = {}

# Knobs:

knob_k = 5 #Parameter for squeeze
norm = 1.0 #What we normalize everything by before applying lambda offset
#norm = "Lambda"
k_ps = 1.5 #Knob problem scale. This is what we multiple times the squeezing function
Dij_knob = True
oneoverDij_knob = False
squeeze = True
if oneoverDij_knob and Dij_knob:
    lambert = "Dij + 1/Dij"
elif Dij_knob and not oneoverDij_knob:
    lambert = "Dij"
elif oneoverDij_knob and not Dij_knob:
    lambert = "1/Dij"
lambert = 1 #This is an override if we want to use paper lambda
configText = configText + "\nDij Term used: " + str(Dij_knob)
configText = configText + "\n1/Dij Term used: " + str(oneoverDij_knob)
configText = configText + "\nknob k: " + str(knob_k)

def calcDist(input1, input2):
    distance = trackList[input1] - trackList[input2]
    totalError = math.sqrt((errorList[input1]) ** 2. + (errorList[input2]) ** 2.)
    return math.fabs(distance / totalError)

def makeQUBO(points, howManyClusters):
    start = time.process_time()
 #Need to find the max distance between any 2 given points. This just iterates to grab all those distances

    maxDist = 0
    avgDij = 0
    aveMinusDij = 0
    dists = []
    for i in range(len(trackList)):
      for j in range(len(trackList)):
        Dij = calcDist(i, j)
        
        if Dij > maxDist:
            maxDist = Dij
    print(maxDist)
    for i in range(len(trackList)):
      for j in range(len(trackList)):
        dists.append(calcDist(i, j) / maxDist)
    dists = np.array(dists)
    avgDij = np.mean(dists)
    dists = 1 - dists
    avgMinusDij = np.mean(dists)
 #This set of loops represents the first term of sigmas in the Hamiltonian
    if Dij_knob:
        j = 0
        while j < len(points):
            i = j + 1 #This is so we dont get a distance of 0 because if i=j then we get i-i which = 0
            while i < len(points):    
                k = 0
                while k < howManyClusters:
                    x = (k*(len(points))+i)
                    y = (k*(len(points))+j)
                    Q[(y,x)] = calcDist(i, j) / maxDist
                    k += 1
                i += 1
            j += 1
    
 #This set of loops represents the second term of sigmas in the Hamiltonian
    if oneoverDij_knob:
        j = 0
        while j < len(points):
            i = j + 1 #This is so we dont get a distance of 0 because if i=j then we get i-i which = 0
            while i < len(points):
                k = 0
                while k < howManyClusters:
                    m = k + 1
                    while m < howManyClusters:
                        x_m = (m*(len(points))+i) 
                        y_m = (m*(len(points))+j)
                        x_k = (k*(len(points))+i) 
                        y_k = (k*(len(points))+j)

                        Q[(y_m,x_k)] = 1 - (calcDist(i, j) / maxDist)
                        Q[(y_k,x_m)] = 1 - (calcDist(i, j) / maxDist)
                        m += 1
                    k += 1
                i += 1
            j += 1
    
 #Now we have to squeeze 
    
    for i in Q.keys():
        Q[i] = Q[i] / norm 
        if squeeze:    
            Q[i] = k_ps * (1 - math.exp(-knob_k * Q[i]))
    
 #This is adding the biases to make sure each point only picks one cluster. It is the thrid group of sigmas in the Hamiltonian
    print(f"lambda is {Lambda()}")
    fudgeFactor = 1.1 #Constraint is only satisfied when this number is greaeter than 1 because we divided by lamnda here
    i = 0
    while i < len(points):
        k = 0
        while k < howManyClusters:
            x = (k*(len(points))+i)
            Q[(x,x)] = -1 * (fudgeFactor * Lambda()) / norm
            k += 1
        m = 0
        while m < howManyClusters:
            l = m + 1
            while l < howManyClusters:
                x = (m*(len(points))+i)
                y = (l*(len(points))+i)
                Q[(x,y)] = 2 * (fudgeFactor * Lambda()) / norm
                l += 1
            m += 1
        i += 1

         #Now we have to squeeze 

    print(f"It took {time.process_time() - start:.2f} seconds to make the QUBO")

def cluster(reads, offsetList, useOffsets):
    print("Started Annealing on the QPU")
    target_Q = dwave.embedding.embed_qubo(Q, embedding, sampler.adjacency, chain_strength=1.0) #Takes my logical QUBO and turns it into a physical QUBO
    if not useOffsets: #Easy way to use offsets or not based on an input to the function
        print("Clustering without offsets...")
        embedded_response = sampler.sample_qubo(target_Q, num_reads=reads, answer_mode="raw")
    else:
        print("Clustering with offsets...")
        embedded_response = sampler.sample_qubo(target_Q, num_reads=reads, answer_mode="raw", anneal_offsets=offsetList) 
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    response = dwave.embedding.unembed_sampleset(embedded_response, embedding, bqm)
    return response

def tabulate(data, printBool, name): #Essentially counts the amount of time that a specific response was returned from the QPU
    tabulatedData = {}
    for i in data:
        if str(i) in tabulatedData.keys():
            tabulatedData[str(i)] += 1
        else:
            tabulatedData[str(i)] = 1
    outfile = open(name, "w")
    json.dump(tabulatedData, outfile)
    outfile.close()
    print(tabulatedData)
    if printBool:
        print(str(data))
    return tabulatedData

def createOffsets(sols):
    logical_j = [0] * (numOfClusters * len(trackList))
    logical_h = [0] * (numOfClusters * len(trackList))
    mef = []
    offsets = [] 
    #offsets = [0] * len(sampler.properties['anneal_offset_ranges'])
    

    for i in Q.keys(): #Initialize the j list with lists. This essentially is a matrix with MATLAB-like indexing
        logical_j[i[0]] = []
    for i in Q.keys(): #This fills the matrix with the row being the qubit and the column being the coupling between the row number and column number
        if i[0] == i[1]:
            logical_h[i[0]] = Q[i]
        else:
            while (max(i)+1) > len(logical_j[min(i)]):
                logical_j[min(i)].append(0)
            logical_j[min(i)][max(i)] = Q[i]
    for i in logical_j[0]:
        logical_j[len(logical_j) - 1].append(0)
    
    logical_j = np.matrix(logical_j)
    logical_j = logical_j.T + logical_j
    
    pd.DataFrame(logical_j).to_csv("/workspace/PHYS324/j.csv")

    for i in range(len(sols)): #Iterating through all the solutions

        mef.append(logical_j.tolist()) #makes mef into a 3D matrix by appending a list of lists to a list

        #finding the "force" on the qubits with just logical j
        for j in range(len(mef[i])): #iterating through the rows of logical j mef[i] is a 2D matrix
            for k in range(len(mef[i][j])): #iterating through the columns logical j mef[i][j] is a 1D list
                mef[i][j][k] *= sols[i][k] #Now this is a field (After it's summed) mef[i][j][k] is a single value 
        for j in range(len(mef[i])): #iterating through the rows of logical j mef[i] is a 2D matrix
            mef[i][j] = sum(mef[i][j]) * int(sols[i][j]) #Now this is a force
            #this turns mef into a 2D matrix
        #Finding the force on the qubits from logical h. We add this to the existing mef
        for j in range(len(logical_h)):
            mef[i][j] = mef[i][j] + (logical_h[j] * -sols[i][j]) #Combining logical j and h
    
    #finding the mean effective force
    mef = np.matrix(mef)
    mef = (mef.mean(0)).tolist()[0]
    
    #Basically parsing this giant dictionary into just lists
    offset_table = sampler.properties['anneal_offset_ranges'] #I think this should return a dictionary with 2 keys and 2 lists
    offset_max = [] #anneal_offset_ranges is the key in the dict. The values is a list of lists
    offset_min = []
    for i in range(len(offset_table)): #Making 2 lists of length 5760 
        offset_min.append(offset_table[i][0])
        offset_max.append(offset_table[i][1])
    for i in range(len(offset_max)):
        if offset_max[i] == 0:
            offset_max[i] = 999
    for i in range(len(offset_min)):
        if offset_min[i] == 0:
            offset_min[i] = -999

    max_offsets = min(np.abs(np.array(offset_max))) #This is the minimum of the maximum offsets. this is possibly a good starting point for what to offset something by. We may need to truncate this later
    min_offsets = max(offset_min)

    #normalize
    mef = np.abs(np.array(mef))
    maxmef = max(mef)
    norm_mef = np.array(mef) / maxmef

    #Map mef values to valid offset for the qubits
    logicalOffsets = (-norm_mef * (max_offsets - min_offsets) + max_offsets).tolist()
    #Applying that to the physical qubit
    offsets = [0.0] * 5760
    for i in range(len(logicalOffsets)): #iterating through logical qubits
        for j in embedding[i]: #Finding all the physical qubits that make up each logical qubit
            offsets[j] = logicalOffsets[i] #applying the offset for a logical qubit to each physical qubit that make up the logical one
    
    """
    logical_J = coo_matrix(
        (-np.ones(u.shape[0]), (u, v)),
        shape=(3, 3)
        )
    logical_h = np.array([[0, 0.9, -1]]).T
    sols = np.array(dw2x_output_baseline['solutions']).T
    sols = reshape_solutions(sols, embeddings)
    dist = calculate_solution_distribution(sols)
    delta = -(sols.T * (logical_J + logical_J.T)).T * sols 
    delta = delta - logical_h * sols
    mean_effective_field = np.mean(np.abs(delta.T), axis=0)
    norm_mef = mean_effective_field / np.max(mean_effective_field)  # normalize
    """
    return offsets

def printQubitGraph(): 
    #name = str(numOfClusters) + "V" + str(len(trackList)) + "T_" + computer
    plt.figure(figsize=(40,40))
    dnx.draw_pegasus_embedding(hardware_graph, emb=embedding, node_size=100, width=2, unused_color=(0,0,0,.25))
    plt.savefig(embedding_filename + ".png") 
    plt.close()

def checkembedding():
    if os.path.exists((embedding_filename + ".json")):
        print("Found existing embedding!")
        embedding_file = open((embedding_filename + ".json"))
        embedding = json.load(embedding_file)
        embedding = {int(k): v for k, v in embedding.items()}
    else:
        print(f'A {probTitle} problem should take about {timeEstimate:.2f} minutes to embed')
        embedding = dwave.embedding.pegasus.find_clique_embedding((len(trackList) * numOfClusters), target_graph=hardware_graph)
        outfile = open((embedding_filename + ".json"), "w")
        json.dump(embedding, outfile)
        outfile.close() 
    return embedding
"""
def Lambda():
    returnme = 0
    if lambert == "Dij + 1/Dij":
        returnme = ((len(trackList) - numOfClusters) + (len(trackList) - 1))
        #returnme = avgDij + avgMinusDij
    elif lambert == "Dij":
        returnme = (len(trackList) - numOfClusters) #This is for when we just use the Dij term
        #returnme = avgDij
    elif lambert == "1/Dij":
        returnme = len(trackList) - 1
        #returnme = avgMinusDij
    else:
        returnme = lambert
    return returnme
"""
def accuracy(sols, reads): #HARDCODED FOR 2 CLUSTERS AS OF NOW
    ncorrect = 0
    answer1 = str((int(len(trackList) / 2) * [0]) + (int(len(trackList) / 2) * [1]) + (int(len(trackList) / 2) * [1]) + (int(len(trackList) / 2) * [0])).replace(",", "")
    answer2 = str((int(len(trackList) / 2) * [1]) + (int(len(trackList) / 2) * [0]) + (int(len(trackList) / 2) * [0]) + (int(len(trackList) / 2) * [1])).replace(",", "")
    for i in list(sols):
        if str(i) == answer1:
            ncorrect += 1
        elif str(i) == answer2:
            ncorrect += 1
    e = ncorrect / reads
    delta = 100 * math.sqrt(e * (1 - e) * reads) / reads
    accu = f"{e * 100}% were correct with an uncertanty of {delta}"
    print(accu)
    return accu

def serialize_response(response, file_name, path):
    serializableResponse = []
    for i in range(len(response.data_vectors['energy'])):
        index = i
        energy = response.data_vectors['energy'][index]
        solution = str(response.record.sample[index])[1:-1].split()
        serializableResponse.append((energy, solution))
    outputFile = open(path + file_name + ".json", "w")
    json.dump(serializableResponse, outputFile)

    sols = response.record.sample
    tabulate(sols, True, (embedding_filename + "_sols.json"))
    accu = accuracy(sols, reads)
    with open(path + "accuracy.txt", "w") as f:
        f.write(accu)
    with open(path + "config.txt", "w") as f:
        f.write(configText)

def CPU(reads, sweeps):
    print("Started the simulated annealing")
    sampler = dimod.SimulatedAnnealingSampler()
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    response = sampler.sample(bqm, num_reads=reads, num_sweeps=sweeps)
    return response

def makeFolder():
    runID = 1
    while True:
        try:
            os.mkdir(probTitle + "_" + str(runID))
            break
        except:
            runID += 1
    return runID

def calc_optimal_lambda(zT_i, zT_unc_i, nV):
    max_dij = 0
    for i in range(len(zT_i)):
        for j in range(i, len(zT_i)):
            dij = abs(zT_i[i] - zT_i[j]) / math.sqrt(zT_unc_i[i] ** 2 + zT_unc_i[j] ** 2)
            if dij > max_dij:
                max_dij = dij
    
    optimal_lambdas = []
    
    for k in range(nV):
        num_tracks_per_cluster = int(len(zT_i) / nV)
    
        for i in range(num_tracks_per_cluster):
            coupling_sum = 0
            for j in range(num_tracks_per_cluster):
                if i == j:
                    continue
                offset = k * num_tracks_per_cluster
                dij = (abs(zT_i[i + offset] - zT_i[j + offset]) / math.sqrt(zT_unc_i[i + offset] ** 2 + zT_unc_i[j + offset] ** 2)) / max_dij
                coupling_sum += 1.5 * (1 - math.exp(-1 * knob_k * dij))
            optimal_lambdas.append(coupling_sum)
    
    return max(optimal_lambdas)
"""
def Lambda():
    #return 1.5 * calc_optimal_lambda(trackList, errorList, numOfClusters)
    return 2.0
"""

def Lambda():
    return 2.2 * calc_optimal_lambda(trackList, errorList, numOfClusters)

#===============================================FINAL FUNCTION CALLS==========================================#

if __name__ == "__main__":
    if norm == "Lambda":
        configText = configText + "\nNormalized by: " + str(norm)
        norm = Lambda()
    else:
        configText = configText + "\nNormalized by: Nothing" 
    makeQUBO(trackList, numOfClusters) #This generates a Hamiltonian with the error adjusted points, the amount of cluster specified at the beginning, and with a stregth of constraint of 1
    print("Started final solve...")

    runID = makeFolder()
    path = "/workspace/PHYS324/" + probTitle + "_" + str(runID) + "/" #Making the folder for the outputs. Starts at runID 1 and uses recursion to find the next run number
    #path = "/workspace/PHYS324/2V20T_1"
    print(path)

    reads = 100
    sweeps = 2000
    if runMode == "QPU":
        embedding = checkembedding() #either grabs embedding from a json, or calculates it using pegasus
        response = cluster(reads, [0], False) #This clusters and stores the answers. Uses a placeholder list for offsets, but we arent using the offsets anyway
        configText = configText + "\nRun Mode: QPU"
        configText = configText + "\nReads: " + str(reads)
    else:
        response = CPU(reads, sweeps) #This uses simulated annealing instead of Quantum Computer Time
        configText = configText + "\nRun Mode: CPU"
        configText = configText + "\nReads: " + str(reads) + ", Sweeps: " + str(sweeps)
    configText = configText + "\nSqueezing: " + str(squeeze)
    configText = configText + "\nLambda: " + str(Lambda())
    if lambert == "Dij + 1/Dij":
        configText = configText + ", (nT - nV) + (nT - 1)(nV - 1)"
    elif lambert == "Dij":
        configText = configText + ", nT - nV" #This is for when we just use the Dij term
    elif lambert == "1/Dij":
        configText = configText + ", (nT - 1)(nV - 1)"
    else:
        configText = configText + ", input constant"
    configText = configText + "\nKnob Problem Scale: " + str(k_ps)
    print("Accuracy without offsets")
    serialize_response(response, "SerializedResponse", path)
        #accuracy(sols, reads)
        #tabulate(sols, False, (embedding_filename + "_sols.json"))

    #offsets = createOffsets(sols)
    #sols = cluster(reads, offsets, True)
    #tabsols = tabulate(sols, False, (embedding_filename + "_sols_AF.json")) #Tabulated solutions
    #print("Accuracy after applying offsets")
    #accuracy(sols, reads)

    #This tabulates the results from the QPU with a specified number of reads. It also takes the offsets that were created by the createOffsets function. 
    #The first bool is for if the offsets should be used or not. The second is for if it should print the raw output from the QPU (untabulated)
    #tabulate(cluster(1000, createOffsets(), False), False)
    #printQubitGraph() #Uncomment this if you want the code to save the image.
    print(configText)