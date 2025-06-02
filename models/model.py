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
from utils.layers import ContextAttentionLayer, dense_layer
from utils.utils import get_device, get_log_molar
from utils.DrugEmbedding import DrugEmbeddingModel

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Main Model
class PASO_GEP_CNV_MUT(nn.Module):
    def __init__(self, params, *args, **kwargs):
        super(PASO_GEP_CNV_MUT, self).__init__(*args, **kwargs)

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
        self.smiles_attention_size = params.get('smiles_attention_size', 64)
        self.gene_attention_size = params.get('gene_attention_size', 1)
        self.molecule_temperature = params.get('molecule_temperature', 1.)
        self.gene_temperature = params.get('gene_temperature', 1.)

        # Model Architecture (Hyperparameters)
        self.molecule_gep_heads = params.get('molecule_gep_heads', [2])
        self.molecule_cnv_heads = params.get('molecule_cnv_heads', [2])
        self.molecule_mut_heads = params.get('molecule_mut_heads', [2])
        self.gene_heads = params.get('gene_heads', [1])
        self.cnv_heads = params.get('cnv_heads', [1])
        self.mut_heads = params.get('mut_heads', [1])
        self.n_heads = params.get('n_heads', 1)
        self.num_layers = params.get('num_layers', 2)
        self.omics_dense_size = params.get('omics_dense_size', 128)
        self.hidden_sizes = (
            [
                # Only use DrugEmbeddingModel output
                self.molecule_gep_heads[0] * params['smiles_embedding_size'] + # 6x256
                self.molecule_cnv_heads[0] * params['smiles_embedding_size'] +
                self.molecule_mut_heads[0] * params['smiles_embedding_size'] +
                sum(self.gene_heads) * self.omics_dense_size +
                sum(self.cnv_heads) * self.omics_dense_size +
                sum(self.mut_heads) * self.omics_dense_size
            ] + params.get('stacked_dense_hidden_sizes', [1024, 512])
        )

        self.dropout = params.get('dropout', 0.5)
        self.temperature = params.get('temperature', 1.)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get('activation_fn', 'relu')]

        # Drug Embedding Model
        bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        self.drug_embedding_model = DrugEmbeddingModel(
            input_feature_dim=78,
            embed_dim=params['smiles_embedding_size'],
            num_heads=8,
            ff_dim=2048,
            num_layers=6,
            num_bond_types=4,
            seq_len=self.smiles_padding_length
        ).to(self.device) #[B, 128, hidden_dim]

        # Attention Layers (single layer from embedding output)
        smiles_hidden_sizes = [params['smiles_embedding_size']]

        self.molecule_attention_layers_gep = nn.Sequential(OrderedDict([
            (
                f'molecule_gep_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[0],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_pathways,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.molecule_temperature
                )
            ) for head in range(self.molecule_gep_heads[0])
        ]))

        self.molecule_attention_layers_cnv = nn.Sequential(OrderedDict([
            (
                f'molecule_cnv_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[0],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_pathways,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.molecule_temperature
                )
            ) for head in range(self.molecule_cnv_heads[0])
        ]))

        self.molecule_attention_layers_mut = nn.Sequential(OrderedDict([
            (
                f'molecule_mut_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[0],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_pathways,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.molecule_temperature
                )
            ) for head in range(self.molecule_mut_heads[0])
        ]))

        self.gene_attention_layers = nn.Sequential(OrderedDict([
            (
                f'gene_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_pathways,
                    context_hidden_size=smiles_hidden_sizes[0],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.gene_temperature
                )
            ) for head in range(self.gene_heads[0])
        ]))

        self.cnv_attention_layers = nn.Sequential(OrderedDict([
            (
                f'cnv_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_pathways,
                    context_hidden_size=smiles_hidden_sizes[0],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.gene_temperature
                )
            ) for head in range(self.cnv_heads[0])
        ]))

        self.mut_attention_layers = nn.Sequential(OrderedDict([
            (
                f'mut_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_pathways,
                    context_hidden_size=smiles_hidden_sizes[0],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.gene_temperature
                )
            ) for head in range(self.mut_heads[0])
        ]))

        self.gep_dense_layers = nn.Sequential(OrderedDict([
            (
                f'gep_dense_0_head_{head}',
                dense_layer(
                    self.number_of_pathways,
                    self.omics_dense_size,
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=params.get('batch_norm', True)
                ).to(self.device)
            ) for head in range(self.gene_heads[0])
        ]))

        self.cnv_dense_layers = nn.Sequential(OrderedDict([
            (
                f'cnv_dense_0_head_{head}',
                dense_layer(
                    self.number_of_pathways,
                    self.omics_dense_size,
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=params.get('batch_norm', True)
                ).to(self.device)
            ) for head in range(self.cnv_heads[0])
        ]))

        self.mut_dense_layers = nn.Sequential(OrderedDict([
            (
                f'mut_dense_0_head_{head}',
                dense_layer(
                    self.number_of_pathways,
                    self.omics_dense_size,
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=params.get('batch_norm', True)
                ).to(self.device)
            ) for head in range(self.mut_heads[0])
        ]))

        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get('batch_norm', True)
                        ).to(self.device)
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = (
            nn.Linear(self.hidden_sizes[-1], 1)
            if not params.get('final_activation', False) else nn.Sequential(
                OrderedDict(
                    [
                        ('projection', nn.Linear(self.hidden_sizes[-1], 1)),
                        ('sigmoidal', ACTIVATION_FN_FACTORY['sigmoid'])
                    ]
                )
            )
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

        gep = torch.unsqueeze(gep, dim=-1)  # [bs, number_of_genes, 1]
        cnv = torch.unsqueeze(cnv, dim=-1)  # [bs, number_of_genes, 1]
        mut = torch.unsqueeze(mut, dim=-1)  # [bs, number_of_genes, 1]
        gep = gep.to(device=self.device)
        cnv = cnv.to(device=self.device)
        mut = mut.to(device=self.device)
        
        # Drug Embedding
        embedded_smiles = self.drug_embedding_model(x, adj_matrix, bond_info)
        embedded_smiles = embedded_smiles[:, 1:, :]

        # Validate output shape
        if embedded_smiles.shape[1] != self.smiles_padding_length or \
           embedded_smiles.shape[2] != self.params['smiles_embedding_size']:
            raise ValueError(
                f"Drug embedding output shape {embedded_smiles.shape} does not match "
                f"expected ([bs, {self.smiles_padding_length}, {self.params['smiles_embedding_size']}])"
            )

        # Use only the embedding output
        encoded_smiles = [embedded_smiles]

        # Molecule context attention
        encodings, smiles_alphas_gep, smiles_alphas_cnv, smiles_alphas_mut = [], [], [], []
        gene_alphas, cnv_alphas, mut_alphas = [], [], []
        for head in range(self.molecule_gep_heads[0]):
            e, a = self.molecule_attention_layers_gep[head](encoded_smiles[0], gep)
            encodings.append(e)
            smiles_alphas_gep.append(a)

        for head in range(self.molecule_cnv_heads[0]):
            e, a = self.molecule_attention_layers_cnv[head](encoded_smiles[0], cnv)
            encodings.append(e)
            smiles_alphas_cnv.append(a)

        for head in range(self.molecule_mut_heads[0]):
            e, a = self.molecule_attention_layers_mut[head](encoded_smiles[0], mut)
            encodings.append(e)
            smiles_alphas_mut.append(a)

        # Gene context attention
        for head in range(self.gene_heads[0]):
            e, a = self.gene_attention_layers[head](gep, encoded_smiles[0], average_seq=False)
            e = self.gep_dense_layers[head](e)
            encodings.append(e)
            gene_alphas.append(a)

        for head in range(self.cnv_heads[0]):
            e, a = self.cnv_attention_layers[head](cnv, encoded_smiles[0], average_seq=False)
            e = self.cnv_dense_layers[head](e)
            encodings.append(e)
            cnv_alphas.append(a)

        for head in range(self.mut_heads[0]):
            e, a = self.mut_attention_layers[head](mut, encoded_smiles[0], average_seq=False)
            e = self.mut_dense_layers[head](e)
            encodings.append(e)
            mut_alphas.append(a)

        encodings = torch.cat(encodings, dim=1) # (bs, 6 x 256, omics_dense x 3)

        # Apply batch normalization if specified
        inputs = self.batch_norm(encodings) if self.params.get('batch_norm', False) else encodings
        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
        prediction_dict = {}

        if not self.training:
            smiles_attention_gep = torch.cat([torch.unsqueeze(p, -1) for p in smiles_alphas_gep], dim=-1)
            smiles_attention_cnv = torch.cat([torch.unsqueeze(p, -1) for p in smiles_alphas_cnv], dim=-1)
            smiles_attention_mut = torch.cat([torch.unsqueeze(p, -1) for p in smiles_alphas_mut], dim=-1)
            gene_attention = torch.cat([torch.unsqueeze(p, -1) for p in gene_alphas], dim=-1)
            cnv_attention = torch.cat([torch.unsqueeze(p, -1) for p in cnv_alphas], dim=-1)
            mut_attention = torch.cat([torch.unsqueeze(p, -1) for p in mut_alphas], dim=-1)
            prediction_dict.update({
                'gene_attention': gene_attention,
                'cnv_attention': cnv_attention,
                'mut_attention': mut_attention,
                'smiles_attention_gep': smiles_attention_gep,
                'smiles_attention_cnv': smiles_attention_cnv,
                'smiles_attention_mut': smiles_attention_mut,
                'IC50': predictions,
                'log_micromolar_IC50':
                    get_log_molar(predictions, ic50_max=self.IC50_max, ic50_min=self.IC50_min)
                    if self.min_max_scaling else predictions
            })

        return predictions, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def _associate_language(self, smiles_language):
        if not isinstance(smiles_language, pytoda.smiles.smiles_language.SMILESLanguage):
            raise TypeError(
                f'Please insert a smiles language (object of type '
                f'pytoda.smiles.smiles_language.SMILESLanguage). Given was {type(smiles_language)}'
            )
        self.smiles_language = smiles_language

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
        'smiles_padding_length': 128,
        'smiles_embedding_size': 512,
        'number_of_pathways': 619,
        'smiles_attention_size': 64,
        'gene_attention_size': 1,
        'molecule_temperature': 1.0,
        'gene_temperature': 1.0,
        'molecule_gep_heads': [2],
        'molecule_cnv_heads': [2],
        'molecule_mut_heads': [2],
        'gene_heads': [1],
        'cnv_heads': [1],
        'mut_heads': [1],
        'n_heads': 2,
        'num_layers': 4,
        'omics_dense_size': 256,
        'stacked_dense_hidden_sizes': [1024, 512],
        'dropout': 0.5,
        'temperature': 1.0,
        'activation_fn': 'relu',
        'batch_norm': True,
        'drug_sensitivity_processing_parameters': {
            'parameters': {'max': 100, 'min': 0}
        },
        'loss_fn': 'mse'
    }

    # Instantiate model
    model = PASO_GEP_CNV_MUT(params).to(get_device())
    model.eval()

    # Test with one batch
    for batch in trainloader:
        drug_data, gep_data, cnv_data, mut_data, ic50 = batch
        with torch.no_grad():
            predictions, prediction_dict = model(drug_data, gep_data, cnv_data, mut_data)
        print("=== Output Check ===")
        print(f"Drug data shapes: x={drug_data[0].shape}, adj_matrix={drug_data[1].shape}")
        print(f"GEP shape: {gep_data.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Prediction dict keys: {prediction_dict.keys()}")
        break


# 수정 사항 -> heads 리스트가 늘어나도 오류가 안나도록 해야함, 원래 코드랑 비교분석
# self.hidden_sizes에 대한 부분 분석이 필요해 보임. 해당 부분이 레이어의 갯수에 따라 붙여서 진행하는 것처럼 보임.