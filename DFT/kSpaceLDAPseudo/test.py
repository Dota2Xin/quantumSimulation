from handlePseudopotential import *

def main():
    root, metadata, radialGrid=getPseudo('Cr')
    print(metadata)
    print(radialGrid)
    print(root)
    #getProjectors(root)
    #printTreeStructure(root)
    #getLocalPart(root)
    return 0

if __name__=="__main__":
    main()