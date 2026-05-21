from handlePseudopotential import *

def main():
    root, metadata, radialGrid=getPseudo('C')
    print(metadata)
    return 0

if __name__=="__main__":
    main()