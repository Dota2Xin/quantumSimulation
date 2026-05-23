from handlePseudopotential import *

def main():
    root, metadata, radialGrid=getPseudo('C')
    #print(metadata)
    #print(radialGrid)
    #print(root)
    getProjectors(root)
    #printTreeStructure(root)
    return 0

if __name__=="__main__":
    main()