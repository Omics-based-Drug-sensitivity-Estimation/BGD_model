import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from collections import OrderedDict

import pytoda
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hyperparams import LOSS_FN_FACTORY, ACTIVATION_FN_FACTORY
from utils.CrossAttention import CrossAttentionModule
from utils.utils import get_device, get_log_molar
from utils.DrugEmbedding_v2 import DrugEmbeddingModel

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Main Model
class BGD_model(nn.Module):
    def __init__(self, params, *args, **kwargs):
        super(BGD_model, self).__init__(*args, **kwargs)

        # Model Parameters
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]
        self.min_max_scaling = True if params.get('drug_sensitivity_processing_parameters', {}) != {} else False
        if self.min_max_scaling:
            self.IC50_max = params['drug_sensitivity_processing_parameters']['parameters']['max']
            self.IC50_min = params['drug_sensitivity_processing_parameters']['parameters']['min']

        # Model Inputs
        self.smiles_padding_length = params['smiles_padding_length']
        self.number_of_pathways = params.get('number_of_pathways', 619)

        # Model Architecture (Hyperparameters)
        self.n_heads = params.get('n_heads', 1)
        self.num_layers = params.get('num_layers', 2)

        self.dropout = params.get('dropout', 0.5)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get('activation_fn', 'relu')]

        # Drug Embedding Model
        bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        self.drug_embedding_model = DrugEmbeddingModel(
            input_feature_dim=78,
            embed_dim=params['smiles_embedding_size'],
            num_heads=self.n_heads,
            ff_dim=2048,
            num_layers=self.num_layers,
            num_bond_types=len(bond_types),
            seq_len=self.smiles_padding_length
        ).to(self.device) #[B, 128, hidden_dim]

        # Attention Layers (single layer from embedding output)

        self.cross_attention_model_gep = CrossAttentionModule(
            hidden_dim = params['smiles_embedding_size'],
            num_layers = self.num_layers,
            num_heads = self.n_heads,
            dropout = self.dropout
        )
        self.cross_attention_model_cnv = CrossAttentionModule(
            hidden_dim = params['smiles_embedding_size'],
            num_layers = self.num_layers,
            num_heads = self.n_heads,
            dropout = self.dropout
        )
        self.cross_attention_model_mut = CrossAttentionModule(
            hidden_dim = params['smiles_embedding_size'],
            num_layers = self.num_layers,
            num_heads = self.n_heads,
            dropout = self.dropout
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(params['smiles_embedding_size'] * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, drug_data, gep, cnv, mut):
        """
        Args:
            drug_data (tuple): Contains (x, adj_matrix, bond_info)
                - x (torch.Tensor): SMILES tokens, shape [bs, smiles_padding_length, feature_dim]
                - adj_matrix (torch.Tensor): Adjacency matrix, shape [bs, smiles_padding_length, smiles_padding_length]
                - bond_info (List[Tuple[int, int, str]]): Bond information for each molecule in the batch
            gep (torch.Tensor): Gene expression data, shape [bs, number_of_genes]
            cnv (torch.Tensor): Copy number variation data, shape [bs, number_of_genes]
            mut (torch.Tensor): Mutation data, shape [bs, number_of_genes]

        Returns:
            (torch.Tensor, dict): predictions, prediction_dict
            predictions is IC50 drug sensitivity prediction of shape [bs, 1].
            prediction_dict includes the prediction and attention weights.
        """
        x, adj_matrix, bond_info = drug_data
        x = x.to(self.device)
        adj_matrix = adj_matrix.to(self.device)
        bond_info = bond_info.to(self.device)

        gep = gep.to(device=self.device)
        cnv = cnv.to(device=self.device)
        mut = mut.to(device=self.device)
        
        # Drug Embedding
        embedded_smiles = self.drug_embedding_model(x, adj_matrix, bond_info)

        output_gep, omics_gep = self.cross_attention_model_gep(embedded_smiles, gep)
        output_cnv, omics_cnv = self.cross_attention_model_cnv(embedded_smiles, cnv)
        output_mut, omics_mut = self.cross_attention_model_mut(embedded_smiles, mut)

        output = torch.cat([output_gep, output_cnv, output_mut], dim = 1)
        
        output = self.output_mlp(output)

        predictions = output 
        prediction_dict = {}

        if not self.training:
            prediction_dict.update({
                'IC50': predictions,
                'log_micromolar_IC50':
                    get_log_molar(predictions, ic50_max=self.IC50_max, ic50_min=self.IC50_min)
                    if self.min_max_scaling else predictions
            })
        return output, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def load(self, path, *args, **kwargs):
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        torch.save(self.state_dict(), path, *args, **kwargs)

# Example Usage
if __name__ == "__main__":
    from data.TripleOmics_Drug_Dataset import TripleOmics_Drug_dataset, custom_collate_fn
    from torch.utils.data import DataLoader

    # File paths
    drug_sensitivity_filepath = 'data/10_fold_data/mixed/MixedSet_train_Fold0.csv'
    smiles_filepath = 'data/CCLE-GDSC-SMILES.csv'
    gep_filepath = 'data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = 'data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = 'data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'

    # Dataset
    dataset = TripleOmics_Drug_dataset(
        drug_sensitivity_filepath=drug_sensitivity_filepath,
        smiles_filepath=smiles_filepath,
        gep_filepath=gep_filepath,
        cnv_filepath=cnv_filepath,
        mut_filepath=mut_filepath,
        gep_standardize=True,
        cnv_standardize=True,
        mut_standardize=True,
        drug_sensitivity_min_max=True,
        column_names=('drug', 'cell_line', 'IC50')
    )

    # DataLoader
    batch_size = 4
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Model parameters
    params = {
        'fold': 10,
        'optimizer': "adam",
        'smiles_padding_length': 128,
        'smiles_embedding_size': 128,
        'number_of_pathways': 619,
        'n_heads': 8,
        'num_layers': 6,
        'dropout': 0.2,
        'activation_fn': 'relu',
        'batch_norm': True,
        'drug_sensitivity_processing_parameters': {
            'parameters': {"min": -8.658382, "max": 13.107465}
        },
        'loss_fn': 'mse'
    }

    # Instantiate model
    model = BGD_model(params).to(get_device())
    model.eval()

    # Test with one batch
    for batch in trainloader:
        drug_data, gep_data, cnv_data, mut_data, ic50 = batch
        with torch.no_grad():
            output, prediction_dict = model(drug_data, gep_data, cnv_data, mut_data)
        print("=== Output Check ===")
        print(f"Drug data shapes: x={drug_data[0].shape}, adj_matrix={drug_data[1].shape}")
        print(f"GEP shape: {gep_data.shape}")
        print(f"Predictions shape: {output.shape}")
        break


# 수정 사항 -> heads 리스트가 늘어나도 오류가 안나도록 해야함, 원래 코드랑 비교분석
# self.hidden_sizes에 대한 부분 분석이 필요해 보임. 해당 부분이 레이어의 갯수에 따라 붙여서 진행하는 것처럼 보임.