import xml.etree.ElementTree as ET
from scipy.special import spherical_jn
from scipy.special import sph_harm_y
import numpy as np

#UPDATE TO MATCH WITH THE hbar=m_e=e=Angstrom=1 units we're using
'''
The Unit Converstion Table:
Me:
e^2=1
4pi epsilon_0=1
hbar=1
m_e=1
1 bohr=1 (a0=hbar^2/(m_e e^2))
Unit of energy= Hartree
Pseudo:
e^2=2
hbar=1
4pi epsilon_0=1
m_e=1/2
1 bohr=1 (a0=hbar^2/(m_e e^2))
Unit of energy= Rydberg

1 Hartree=2 Rydberg 
Scale potentials down by 2, scale states not at all.
'''
scaleFactor=1/2

def getProjectors(root):
    nlPP=root.find('PP_NONLOCAL')
    projectors=[]
    angularMomenta=[]
    cutoffs=[]
    D=0
    for child in nlPP:
        if "BETA" in child.tag:
            dict=child.attrib
            angularMomenta.append(dict['angular_momentum'])
            cutoffs.append(dict['cutoff_radius_index'])
            data=np.fromstring(child.text, sep=' ')
            projectors.append(data)
        else:
            data=np.fromstring(child.text, sep=' ')
            D=data*scaleFactor
    D=D.reshape((len(projectors), len(projectors)))

    return projectors, angularMomenta, cutoffs, D

#spline interpolation
def splineInterpolate(q, data):
    return 0

def fillInterpolation(splineData, fillGrid):
    return 0

def getTheta(qGrid):
    '''
    on a given vector we do:
    theta=np.arccos(q_z/|q|)
    '''
    qNorm=np.linalg.norm(qGrid, axis=-1)
    theta=np.arccos(np.divide(qGrid[:,:,:,2], qNorm, out=np.zeros_like(qNorm), where=(qNorm!=0)))
    return theta

def getPhi(qGrid):
    '''
        on a given vector we do:
        phi=np.arctan2(q_y, q_x)
    '''
    phi=np.arctan2(qGrid[:,:,:,1], qGrid[:,:,:,0])
    return phi

def getProjectorVec(projectorIntegral, qGrid, position,l,m):
    expArg=np.einsum('ijkl, l->ijk', qGrid, position)
    pre=4*np.pi*(1j**l)*np.exp(-1j*expArg)
    thetaGrid=getTheta(qGrid)
    phiGrid=getPhi(qGrid)
    pre=pre*sph_harm_y(l,m, thetaGrid, phiGrid)
    return projectorIntegral*pre

def projectorIntegral(r, rab, projector, q, l):
    bessel=spherical_jn(l, np.outer(q,r))
    return np.sum(bessel*projector*rab, axis=-1)

def getLocalPart(root):
    localPP=root.find('PP_LOCAL')
    localPseudo=np.fromstring(localPP.text, sep=' ')
    return localPseudo*scaleFactor

def localIntegral(r, rab, pseudo, gaussian, G, rC):
    radial=pseudo-gaussian(r, rC)
    rest=np.sin(np.outer(G,r))*r
    return np.sum(rest*radial*rab, axis=-1)

#MAYBE HAVE TO SCALE (up or down) BY 4pi*r^2
def getCoreDensity(root):
    xcCC = root.find('PP_NLCC')
    coreDensity = np.fromstring(xcCC.text, sep=' ')
    print(coreDensity)
    return coreDensity

def coreDensityIntegral(coreDensity, r,rab, G):
    integrand=np.sin(np.outer(G,r))*coreDensity*r
    return np.sum(integrand*rab, axis=-1)

#MAYBE HAVE TO SCALE (up or down) BY 4pi*r^2
def getInitialDensity(root):
    valence = root.find('PP_RHOATOM')
    valenceDensity = np.fromstring(valence.text, sep=' ')
    print(valenceDensity)
    return valenceDensity

def initialDensityIntegral(initialDensity,r,rab,G):
    integrand = np.sin(np.outer(G, r)) * initialDensity * r
    return np.sum(integrand * rab, axis=-1)

def getPseudo(element):
    with open('oncvPseudos/'+element+'.upf', 'r', encoding='utf-8') as f:
        xml_content = f.read()

    try:
        # Since it starts directly with <UPF>, we can parse it straight away
        root = ET.fromstring(xml_content.strip())
    except ET.ParseError as e:
        raise ValueError(
            f"XML Parsing failed. Ensure there are no trailing garbage characters at the end of the file. Error: {e}")

        # 1. Extract metadata safely from the header node
    header = root.find("PP_HEADER")
    metadata = {}
    if header is not None:
        metadata = {
            "element": header.get("element").strip(),
            "pseudo_type": header.get("pseudo_type"),
            "core_correction": header.get("core_correction") == "T",
            "mesh_size": int(header.get("mesh_size", 0)),
            "number_of_projectors": int(header.get("number_of_proj", 0))
        }

    # 2. Extract the radial grid mesh points into a NumPy array
    mesh_element = root.find("PP_MESH/PP_R")
    radialGrid = None
    if mesh_element is not None and mesh_element.text:
        # np.fromstring handles any arbitrary spaces or newlines gracefully
        radialGrid = np.fromstring(mesh_element.text, sep=' ')

    return root, metadata, radialGrid


def printTreeStructure(element, depth=0):
    # Create indentation proportional to the depth in the XML tree
    indent = "  " * depth

    # Format the attributes if they exist
    attrs = " ".join([f'{k}="{v}"' for k, v in element.items()])
    attr_str = f" [{attrs}]" if attrs else ""

    # Inspect text content (strip whitespace and check length)
    text_summary = ""
    if element.text and element.text.strip():
        clean_text = element.text.strip()
        # If it's a long data block, summarize it rather than printing raw numbers
        if len(clean_text) > 40:
            text_summary = f" -> (Data Block: {len(clean_text.split())} values)"
        else:
            text_summary = f" -> '{clean_text}'"

    # Print the current node details
    print(f"{indent}<{element.tag}>{attr_str}{text_summary}")

    # Recursively traverse child elements
    for child in element:
        printTreeStructure(child, depth + 1)