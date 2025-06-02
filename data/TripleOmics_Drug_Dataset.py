import os
import deepchem as dc
from rdkit import Chem
import numpy as np

import logging

# DeepChem 로거 가져오기
dc_logger = logging.getLogger("deepchem")
dc_logger.setLevel(logging.ERROR)  # ERROR 이상만 출력

# dev date 2023/11/25 14:28
from numpy import log
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import List, Tuple, Dict


# 커스텀 collate_fn 정의  -> 고정된 패딩 크기로 바꾸는 방식을 사용해야 할 듯
def custom_collate_fn(batch):
    # batch: [(drug_data, gep_data, cnv_data, mut_data, ic50), ...]
    # drug_data: (x, adj_matrix, bond_info)

    # 1. drug_data 분리
    drug_data = [item[0] for item in batch]  # [(x, adj_matrix, bond_info), ...]
    xs = [item[0] for item in drug_data]  # [x1, x2, ...]
    adj_matrices = [item[1] for item in drug_data]  # [adj_matrix1, adj_matrix2, ...]
    bond_matrices = [item[2] for item in drug_data]  # [bond_matrix1, bond_matrix2, ...]

    padded_xs = torch.zeros(len(xs), 128, xs[0].shape[1])  # [batch_size, max_len, feature_dim]
    for i, x in enumerate(xs):
        padded_xs[i, :x.shape[0], :] = x.clone()

    # 3. adj_matrix 패딩
    padded_adj_matrices = torch.zeros(len(adj_matrices), 128, 128)
    for i, adj in enumerate(adj_matrices):
        num_nodes = adj.shape[-1]
        padded_adj_matrices[i, :num_nodes, :num_nodes] = adj.clone()
        
    padded_bond_matrices = torch.zeros(len(bond_matrices), 128, 128)
    for i, bond_matrix in enumerate(bond_matrices):
        num_nodes = bond_matrix.shape[-1]
        padded_bond_matrices[i, :num_nodes, :num_nodes] = bond_matrix.clone()
    # 4. 나머지 데이터 처리
    gep_data = torch.stack([item[1] for item in batch])
    cnv_data = torch.stack([item[2] for item in batch])
    mut_data = torch.stack([item[3] for item in batch])
    ic50 = torch.stack([item[4] for item in batch])

    return (padded_xs, padded_adj_matrices, padded_bond_matrices), gep_data, cnv_data, mut_data, ic50

# 정규화된 인접 행렬 생성
def create_adj_matrix(adj_list: List[List[int]], num_nodes: int) -> torch.Tensor:
    adj = torch.eye(num_nodes)
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            adj[node, neighbor] = 1
            adj[neighbor, node] = 1
    return adj

# def get_bond_info(mol: Chem.Mol) -> list:
#     """RDKit Mol 객체에서 결합 정보 추출."""
#     bond_info = []
#     for bond in mol.GetBonds():
#         start_atom = bond.GetBeginAtomIdx()
#         end_atom = bond.GetEndAtomIdx()
#         bond_type = str(bond.GetBondType())  # 예: SINGLE, DOUBLE, TRIPLE, AROMATIC
#         bond_info.append((start_atom, end_atom, bond_type))
#     return bond_info

def get_bond_matrix(mol: Chem.Mol, num_nodes: int) -> torch.Tensor:
    """RDKit Mol 객체에서 bond_matrix 생성."""
    # bond_types 정의
    bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    bond_type_to_idx = {btype: idx + 1 for idx, btype in enumerate(bond_types)}  # 1, 2, 3, 4로 매핑
    bond_matrix = torch.zeros(num_nodes, num_nodes)
    for bond in mol.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        bond_type = str(bond.GetBondType())
        if bond_type in bond_type_to_idx:
            bond_value = bond_type_to_idx[bond_type]  # SINGLE: 1, DOUBLE: 2, TRIPLE: 3, AROMATIC: 4
            bond_matrix[start_atom, end_atom] = bond_value
            bond_matrix[end_atom, start_atom] = bond_value
    return bond_matrix

class TripleOmics_Drug_dataset(Dataset):
    def __init__(self,
                 drug_sensitivity_filepath: str,
                 smiles_filepath: str, #CCLE-GDSC-SMILES.csv
                 gep_filepath: str,
                 cnv_filepath: str,
                 mut_filepath: str,
                 gep_standardize: bool = False,
                 cnv_standardize: bool = False,
                 mut_standardize: bool = False,
                 drug_sensitivity_dtype: torch.dtype = torch.float,
                 gep_dtype: torch.dtype = torch.float,
                 cnv_dtype: torch.dtype = torch.float,
                 mut_dtype: torch.dtype = torch.float,
                #  smiles_language: SMILESLanguage = None,
                 drug_sensitivity_min_max: bool = True,
                 column_names: Tuple[str] = ['drug', 'cell_line', 'IC50'],
                 ):
        self.drug_sensitivity = pd.read_csv(drug_sensitivity_filepath, index_col=0)
        self.smiles = pd.read_csv(smiles_filepath) 
        self.preprocess_smiles()
        self.gep_standardize = gep_standardize
        self.cnv_standardize = cnv_standardize
        self.mut_standardize = mut_standardize
        self.drug_sensitivity_dtype = drug_sensitivity_dtype
        self.gep_dtype = gep_dtype
        self.cnv_dtype = cnv_dtype
        self.mut_dtype = mut_dtype
        # self.smiles_language = smiles_language
        self.drug_sensitivity_min_max = drug_sensitivity_min_max
        self.drug_sensitivity_processing_parameters = {}
        self.column_names = column_names
        self.drug_name, self.cell_name, self.label_name = self.column_names
        if gep_filepath is not None:
            self.gep = pd.read_csv(gep_filepath, index_col=0)
            if gep_standardize:
                scaler = StandardScaler()
                self.gep_standardized = scaler.fit_transform(self.gep)
                self.gep = pd.DataFrame(self.gep_standardized, index=self.gep.index)
        if cnv_filepath is not None:
            self.cnv = pd.read_csv(cnv_filepath, index_col=0)
            if cnv_standardize:
                scaler = StandardScaler()
                self.cnv_standardized = scaler.fit_transform(self.cnv)
                self.cnv = pd.DataFrame(self.cnv_standardized, index=self.cnv.index)
        if mut_filepath is not None:
            self.mut = pd.read_csv(mut_filepath, index_col=0)
            if mut_standardize:
                scaler = StandardScaler()
                self.mut_standardized = scaler.fit_transform(self.mut)
                self.mut = pd.DataFrame(self.mut_standardized, index=self.mut.index)

        # NOTE: optional min-max scaling
        if self.drug_sensitivity_min_max:
            minimum = self.drug_sensitivity_processing_parameters.get(
                'min', self.drug_sensitivity[self.label_name].min()
            )
            maximum = self.drug_sensitivity_processing_parameters.get(
                'max', self.drug_sensitivity[self.label_name].max()
            )
            self.drug_sensitivity[self.label_name] = (
                self.drug_sensitivity[self.label_name] - minimum
            ) / (maximum - minimum)
            self.drug_sensitivity_processing_parameters = {
                'processing': 'min_max',
                'parameters': {'min': minimum, 'max': maximum},
            }

    def preprocess_smiles(self):
            self.smiles_features = {}
            featurizer = dc.feat.graph_features.ConvMolFeaturizer(use_chirality=True)
            for _, row in self.smiles.iterrows():
                drug = row["DRUG_NAME"]
                smiles = row["SMILES"]
                mol = Chem.MolFromSmiles(smiles)
                mol_object = featurizer.featurize([mol])[0]
                features = torch.from_numpy(mol_object.atom_features)
                adj_list = mol_object.canon_adj_list
                adj_matrix = create_adj_matrix(adj_list, len(adj_list))
                bond_matrix = get_bond_matrix(mol, len(adj_list))  # bond_matrix 생성
                self.smiles_features[drug] = (features, adj_matrix, bond_matrix)

    def __len__(self):
        return len(self.drug_sensitivity)

    def __getitem__(self, index):
        # drug sensitivity
        # molecules = []
        selected_sample = self.drug_sensitivity.iloc[index]
        selected_drug = selected_sample[self.drug_name]
        ic50_tensor = torch.tensor(
            [selected_sample[self.label_name]],
            dtype=self.drug_sensitivity_dtype,
        )
        features, adj_matrix, bond_matrix = self.smiles_features[selected_drug]
        
        # omics data
        gene_expression_tensor = torch.tensor((
            self.gep.loc[selected_sample[self.cell_name]].values),
            dtype=self.gep_dtype)
        cnv_tensor = torch.tensor((
            self.cnv.loc[selected_sample[self.cell_name]].values),
            dtype=self.cnv_dtype)
        mut_tensor = torch.tensor((
            self.mut.loc[selected_sample[self.cell_name]].values),
            dtype=self.mut_dtype)
        return ([features, adj_matrix, bond_matrix], gene_expression_tensor,
                cnv_tensor, mut_tensor, ic50_tensor)


if __name__ == "__main__":
    # 파일 경로 정의 (지정한 디렉토리에 파일이 있다고 가정)
    drug_sensitivity_filepath = 'data/10_fold_data/mixed/MixedSet_test_Fold0.csv'  # 예시 폴드 파일
    smiles_filepath = 'data/CCLE-GDSC-SMILES.csv'
    gep_filepath = 'data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = 'data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = 'data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'

    # 데이터셋 초기화
    dataset = TripleOmics_Drug_dataset(
        drug_sensitivity_filepath=drug_sensitivity_filepath,
        smiles_filepath=smiles_filepath,
        gep_filepath=gep_filepath,
        cnv_filepath=cnv_filepath,
        mut_filepath=mut_filepath,
        gep_standardize=True,  # 유전자 발현 데이터 표준화
        cnv_standardize=True,  # CNV 데이터 표준화
        mut_standardize=True,  # 돌연변이 데이터 표준화
        drug_sensitivity_min_max=True,  # IC50 값 min-max 정규화
        column_names=('drug', 'cell_line', 'IC50')  # 컬럼 이름 설정
    )

    # 데이터셋 길이 확인
    print(f"데이터셋 크기: {len(dataset)}")

    # 첫 번째 샘플 가져오기 및 확인
    sample = dataset[10]
    drug_data, gep_data, cnv_data, mut_data, ic50 = sample

    # max = 0
    # for a in dataset:
    #     drug_data, gep_data, cnv_data, mut_data, ic50 = a
    #     if max < drug_data[0].shape[0]:
    #         max = drug_data[0].shape[0]
    print(max)
    # 샘플 데이터 구조 출력
    print("\n샘플 데이터 구조:")
    print(f"약물 데이터 (features, adj_list, degree_list, bond_info):")
    print(f" - Atom features shape: {drug_data[0].shape}")
    print(f" - Adjacency list length: {drug_data[1].shape}")
    print(f" - Bond info length: {len(drug_data[2])}")
    print(f"유전자 발현 데이터 shape: {gep_data.shape}")
    print(f"CNV 데이터 shape: {cnv_data.shape}")
    print(f"돌연변이 데이터 shape: {mut_data.shape}")
    print(f"IC50 값: {ic50.item()}")

    # 데이터셋의 첫 5개 샘플 반복하며 IC50 값 확인
    print("\n첫 5개 샘플의 IC50 값:")
    for i in range(min(5, len(dataset))):
        _, _, _, _, ic50 = dataset[i]
        print(f"샘플 {i}: IC50 = {ic50.item()}")