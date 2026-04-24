from upf_tools import UPFDict

#we'll fill it up with pseudopotentials from https://pseudopotentials.quantum-espresso.org/legacy_tables
def initUPF(element):
    post='.pbe-n-kjpaw_psl.1.0.0.UPF'
    dict=UPFDict.from_upf('pseudopotentials/'+element+post)
    print(dict)
    for key in dict:
        print(key)
    print(dict['mesh'])
    print(getRadius(dict))
    print(getRadiusDiff(dict))
    print("--------BREAK--------")
    print(dict['pswfc'])
    print(dict['pwsfc']['chi'])
    print("--------BREAK--------")

    print(dict['paw'])

def getRadius(dict):
    return dict['mesh']['r']

def getRadiusDiff(dict):
    return dict['mesh']['rab']

