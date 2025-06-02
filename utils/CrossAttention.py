import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 선형 변환
        Q = self.W_q(query)  # (B, seq_len_q, d_model)
        K = self.W_k(key)    # (B, seq_len_k, d_model)
        V = self.W_v(value)  # (B, seq_len_k, d_model)
        
        # 헤드 분할
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, seq_len_q, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, seq_len_k, d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, seq_len_k, d_k)
        
        # 스케일드 닷-프로덕트 어텐션
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, num_heads, seq_len_q, seq_len_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # (B, num_heads, seq_len_q, d_k)
        
        # 헤드 결합
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (B, seq_len_q, d_model)
        output = self.W_o(context)  # (B, seq_len_q, d_model)
        
        return output

class CustomTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(CustomTransformerLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        # 크로스 어텐션
        attn_output = self.attention(query, key, value)
        query = self.norm1(query + self.dropout1(attn_output))
        
        # 피드포워드
        ffn_output = self.ffn(query)
        output = self.norm2(query + self.dropout2(ffn_output))
        
        return output

class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, num_heads=8, dropout=0.1):
        super(CrossAttentionModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CLS 토큰 초기화
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Omics 데이터 처리를 위한 MLP (619 -> 256)
        self.omics_mlp = nn.Sequential(
            nn.Linear(619, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 커스텀 트랜스포머 레이어
        self.transformer_layers_drug = nn.ModuleList([
            CustomTransformerLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.transformer_layers_omics = nn.ModuleList([
            CustomTransformerLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        
    def forward(self, drug_embedding, omics_data):
        batch_size = drug_embedding.size(0)
        
        # CLS 토큰 추가
        cls_token = self.cls_token.expand(batch_size, 1, self.hidden_dim)
        drug_embedding = torch.cat([cls_token, drug_embedding], dim=1)  # (B, 129, 256)
        
        # Omics 데이터 처리
        omics = self.omics_mlp(omics_data).unsqueeze(1)  # (B, 1, 256)
        
        # 트랜스포머 레이어 통과
        output_drug = drug_embedding
        output_omics = omics
        
        for layer1, layer2 in zip(self.transformer_layers_drug, self.transformer_layers_omics):
            output_drug = layer1(query=output_drug, key=output_omics, value=output_omics)  # (B, 129, 256)
            output_omics = layer2(query=output_omics, key=output_drug, value=output_drug)
        
        # CLS 토큰 추출 및 IC50 예측
        cls_output = output_drug[:, 0, :]  # (B, 256)
        omics_output = output_omics[:, 0, :]
        
        return cls_output, omics_output

# 사용 예시
if __name__ == "__main__":
    # 입력 데이터 예시
    batch_size = 32
    drug_embedding = torch.randn(batch_size, 128, 256)
    omics_data = torch.randn(batch_size, 619)
    
    # 모델 초기화
    model = CrossAttentionModule(hidden_dim=256, num_layers=2, num_heads=8, dropout=0.1)
    
    # 예측
    ic50_pred = model(drug_embedding, omics_data)
    print(f"Predicted IC50 shape: {ic50_pred.shape}")  # (32, 1)