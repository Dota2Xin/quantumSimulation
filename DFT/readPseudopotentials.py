from upf_tools import UPFDict

#we'll fill it up with pseudopotentials from https://pseudopotentials.quantum-espresso.org/legacy_tables
def testUPF(element):
    post='.pbe-n-kjpaw_psl.1.0.0.UPF'
    dict=UPFDict.from_upf('pseudopotentials/'+element+post)
    print(dict)
    for key in dict:
        print(key)
    #START WORKING ON GETTING PROJECTORS
    print(dict['nonlocal'])
    '''
    print(dict['mesh'])
    print(len(getRadius(dict)))
    print(getRadius(dict))
    print(len(getRadiusDiff(dict)))
    print(getRadiusDiff(dict))
    print("--------BREAK--------")
    print(dict['pswfc'])
    print(dict['pswfc']['chi'][0]['content'])
    print("--------BREAK--------")
    print(dict['full_wfc'])
    #print(dict['paw'])
    '''

def getDict(element):
    post = '.pbe-n-kjpaw_psl.1.0.0.UPF'
    dict = UPFDict.from_upf('pseudopotentials/' + element + post)
    return dict

def getRadius(dict):
    return dict['mesh']['r']

def getRadiusDiff(dict):
    return dict['mesh']['rab']

def getWavefunctionsVE(dict):
    lValues=[]
    waveFuncs=[]
    #put an absolute index on the wavefunctions that encapsulates (n,l,m) and have lDict, nDict
    for waveFunc in dict['pswfc']['chi']:
        lValues.append(waveFunc['l'])
        waveFuncs.append(waveFunc['content'])
    indexDict=[]
    mValues=[]
    for i in range(len(lValues)):
        index=i
        l=lValues[i]
        m=-l
        for j in range(int(2*l+1)):
            mValues.append(m)
            m+=1
            indexDict.append(index)

    return waveFuncs, lValues, mValues, indexDict

def getWavefunctionsAE(dict):
    lValues=[]
    waveFuncs=[]
    #put an absolute index on the wavefunctions that encapsulates (n,l,m) and have lDict, nDict
    labels={}
    for waveFunc in dict['full_wfc']['aewfc']:
        if waveFunc['label'] in labels:
            continue
        else:
            labels[waveFunc['label']]=1
        lValues.append(waveFunc['l'])
        waveFuncs.append(waveFunc['content'])
    indexDict=[]
    mValues=[]
    for i in range(len(lValues)):
        index=i
        l=lValues[i]
        m=-l
        for j in range(int(2*l+1)):
            mValues.append(m)
            m+=1
            indexDict.append(index)

    return waveFuncs, lValues, mValues, indexDict
