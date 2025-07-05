from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def extract_descriptors(mol):
    descriptors = {name: func(mol) for name, func in Descriptors.descList}
    return descriptors


def extract_descriptors_for_all_molecules(smiles_list):
    descriptor_data = []

    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            descriptors = extract_descriptors(mol)
            descriptor_values = np.array(list(descriptors.values()))
            descriptor_data.append(descriptor_values)

    return np.array(descriptor_data)


def get_similar_descriptors(smiles_list, percent=5):

    descriptor_matrix = extract_descriptors_for_all_molecules(smiles_list)


    similarity_matrix = cosine_similarity(descriptor_matrix.T)


    mean_similarities = np.mean(similarity_matrix, axis=0)


    total_descriptors = len(Descriptors.descList)
    top_n = int(np.ceil(total_descriptors * percent / 100))


    top_n_indices = np.argsort(mean_similarities)[::-1][:top_n]


    top_n_descriptors = [(list(Descriptors.descList)[i][0], mean_similarities[i]) for i in top_n_indices]

    return top_n_descriptors


smiles_list = [
    "S(=O)(=O)(C1=CC=C(N)C=C1)C2=CC=C(N)C=C2",
    "O=C1N2C(C=3C(CC2)=CC=CC3)CN(C(=O)C4CCCCC4)C1",
    "CCCSC1=CC2=C(C=C1)N=C(N2)NC(=O)OC",
    "C(OC)(=O)C=1C(C(C(OC)=O)=C(C)NC1C)C2=C(N(=O)=O)C=CC=C2",
    "O(CCCCN1CCN(CC1)C2=C(Cl)C(Cl)=CC=C2)",
    "C(C(O)=O)C1(CC)C2=C(C=3C(N2)=C(CC)C=CC3)CCO1",
    "O(C(C(CC)(C)C)=O)[C@@H]1[C@]2(C(C=C[C@H](C)[C@@H]2CC[C@@H]3C[C@@H](O)CC(=O)O3)=C[C@H](C)C1)[H]",
    "S(CC1=C(C)C(OCC(F)(F)F)=CC=N1)(=O)C=2NC=3C(N2)=CC=CC3",
    "CN1CCN(C=2C=3C(NC=4C(N2)=CC(Cl)=CC4)=CC=CC3)CC1",
    "C1=NNC2=C1C(=O)NC=N2",
    "CC(C)(C(=O)O)OC1=CC=C(C=C1)CCNC(=O)C2=CC=C(C=C2)Cl",
    "N1C2=C(N([H])[H])N=C(N([H])[H])N=C2N=C(C=1C1C([H])=C([H])C([H])=C([H])C=1[H])N([H])[H]",
    "O=C(O)C(N(C(=O)CCCC)CC=1C=CC(=CC1)C=2C=CC=CC2C3=NN=NN3)C(C)C",
    "O=C1[C@@]2(OC=3C1=C(OC)C=C(OC)C3Cl)C(OC)=CC(=O)C[C@H]2C",
    "C(C1=C(Cl)C=CC=C1)(N2C=CN=C2)(C3=CC=CC=C3)C4=CC=CC=C4",
    "C(NCCC1=CC=C(S(NC(NC2CCCCC2)=O)(=O)=O)C=C1)(=O)C3=C(OC)C=CC(Cl)=C3",
    "CC(CS(=O)(=O)C1=CC=C(C=C1)F)(C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F)O",
    "S(=O)(=O)(C1=CC=C(N)C=C1)C2=CC=C(N)C=C2",
    "FC=1C=C2C(C(=NO2)C3CCN(CCC=4C(=O)N5C(=NC4C)CCCC5)CC3)=CC1",
    "CC1=C(SC(=N1)C2=CC(=C(C=C2)OCC(C)C)C#N)C(=O)O",
    "CCCCC1=NC2(CCCC2)C(=O)N1CC3=CC=C(C=C3)C4=CC=CC=C4C5=NNN=N5",
    "C(NCCC1=CC=C(S(NC(N[C@H]2CC[C@H](C)CC2)=O)(=O)=O)C=C1)(=O)N3C(=O)C(CC)=C(C)C3",
    "C1=CC=C(C=C1)C(C2=CC=CC=C2)S(=O)CC(=O)N",
    "CCCN(CCC)S(=O)(=O)C1=CC=C(C=C1)C(=O)O",
    "CCC(C)N1C(=O)N(C=N1)C2=CC=C(C=C2)N3CCN(CC3)C4=CC=C(C=C4)OC[C@H]5CO[C@](O5)(CN6C=NC=N6)C7=C(C=C(C=C7)Cl)Cl",
    "C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N",
    "CC[C@H](C)C(=O)O[C@H]1C[C@H](C=C2[C@H]1[C@H]([C@H](C=C2)C)CC[C@@H]3C[C@H](CC(=O)O3)O)C",
    "CC(C1=CC2=C(C=C1)SC3=CC=CC=C3C(=O)C2)C(=O)O",
    "COC1=CC=CC=C1OCCNCC(COC2=CC=CC3=C2C4=CC=CC=C4N3)O",
    "CCCC1=NC2=C(N1CC3=CC=C(C=C3)C4=CC=CC=C4C(=O)O)C=C(C=C2C)C5=NC6=CC=CC=C6N5C",
]

top_similar_descriptors = get_similar_descriptors(smiles_list, percent=1)

print("Top similar descriptors (1% of total descriptors):")
for descriptor, similarity in top_similar_descriptors:
    print(f"{descriptor}: {similarity}")
