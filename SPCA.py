from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler

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

def select_descriptors_with_sparse_pca(smiles_list, n_components=5, alpha=1.0, zero_threshold=0.9):

    descriptor_matrix = extract_descriptors_for_all_molecules(smiles_list)

    df = pd.DataFrame(descriptor_matrix, columns=[name for name, _ in Descriptors.descList])

    df = df.dropna(axis=1)

    zero_ratio = (df == 0).mean()
    columns_to_drop = zero_ratio[zero_ratio > zero_threshold].index
    df = df.drop(columns=columns_to_drop)

    scaler = StandardScaler()
    descriptor_matrix_standardized = scaler.fit_transform(df)

    spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
    spca.fit(descriptor_matrix_standardized)

    sparse_loadings = spca.components_

    selected_descriptors = set()
    for i in range(n_components):

        max_index = np.argmax(np.abs(sparse_loadings[i]))
        selected_descriptors.add(max_index)

    descriptor_names = [df.columns[i] for i in selected_descriptors]

    return descriptor_names

solvent_smiles_list = [
    "C(CCl)Cl",
    "C1COCCO1",
    "CCCCO",
    "CCC(C)O",
    "CCC(=O)C",
    "CCCCOCCO",
    "CCOCCO",
    "COCCO",
    "CC(C)O",
    "CCCOCCO",
    "CC(=O)C",
    "CC#N",
    "CCCCOC(=O)C",
    "CCO",
    "CO",
    "C(Cl)(Cl)Cl",
    "C1CCC(=O)CC1",
    "C(Cl)Cl",
    "CC(=O)N(C)C",
    "CN(C)C=O",
    "CS(=O)C",
    "C(CO)O",
    "CCOC(=O)C",
    "CCOC=O",
    "CCC(=O)OCC",
    "O",
    "CC(C)CCO",
    "CC(C)CCOC(=O)C",
    "CC(C)CO",
    "CC(C)COC(=O)C",
    "CC(C)OC(=O)C",
    "CC(=O)OC",
    "CC(C)CC(=O)C",
    "CCC(=O)OC",
    "CCCCCOC(=O)C",
    "CCCCCCCO",
    "CCCCCCO",
    "CNC=O",
    "CN1CCCC1=O",
    "CCCCCCCCO",
    "CCCCCO",
    "CCCO",
    "CCCOC(=O)C",
    "C1CCOC1",
    "CC1=CC=CC=C1",
    "C(CCO)CO",
    "C1=CC=CC=C1",
    "C(C(CO)O)O",
    "CC(CO)O",
    "CCCCCC",
    "CCOCCOCCO",
    "CC(C)(C)OC",
    "C1CCCCC1",

]

selected_descriptors = select_descriptors_with_sparse_pca(solvent_smiles_list, n_components=3, alpha=1.0, zero_threshold=0.9)

print("Selected descriptors using Sparse PCA:")
for descriptor in selected_descriptors:
    print(descriptor)