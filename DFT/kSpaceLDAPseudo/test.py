from handlePseudopotential import *

def main():
    root, metadata, radialGrid, rab=getPseudo('C')
    print(metadata)
    #print(radialGrid)
    print(root)
    #print(rab)
    #print(len(rab))
    #print(len(radialGrid))
    #getProjectors(root)
    print(getZ(root))
    #printTreeStructure(root)
    #getLocalPart(root)
    return 0

if __name__=="__main__":
    main()