import xml.etree.ElementTree as ET
import numpy as np

def getProjectors():
    return 0

def actProjectors():
    return 0

def getLocalPart():
    return 0

def getCoreDensity():
    return 0

def handleCoreDensity():
    return 0

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