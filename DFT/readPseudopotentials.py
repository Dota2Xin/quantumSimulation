from upf_tools import UPFDict

#we'll fill it up with pseudopotentials from https://pseudopotentials.quantum-espresso.org/legacy_tables
def initUPF(element):
    post='.pbe-n-kjpaw_psl.1.0.0.UPF'
    dict=UPFDict.from_upf('pseudopotentials/'+element+post)
    print(dict)
    for key in dict:
        print(key)
    print(dict['full_wfc'])
    for key in dict['full_wfc']:
        print(key)
    print(dict['full_wfc']['aewfc'])
    print(dict['full_wfc']['number_of_wfc'])
    print(dict['full_wfc']['pswfc'])