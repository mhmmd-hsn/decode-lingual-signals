import torch
from torch import nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 concat: bool = True, dropout: float = 0.6, leaky_relu_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat  
        self.dropout = dropout
        self.leaky_relu_slope = leaky_relu_slope

        if concat:
            assert out_features % n_heads == 0, "out_features must be divisible by n_heads when concatenating"
            self.head_dim = out_features // n_heads  
        else:
            self.head_dim = out_features

        self.W = nn.Parameter(torch.empty(in_features, self.head_dim * n_heads))
        self.a = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim, 1))
        
        self.leakyrelu = nn.LeakyReLU(self.leaky_relu_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.a, gain=1.414)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        squeeze_output = False
        if h.dim() == 2:
            h = h.unsqueeze(0)       # shape -> (1, N, F)
            adj_mat = adj_mat.unsqueeze(0)  # shape -> (1, N, N)
            squeeze_output = True

        B, N, _ = h.size()

        
        h = F.dropout(h, p=self.dropout, training=self.training)
        Wh = torch.matmul(h, self.W)                     # (B, N, head_dim * n_heads)
        Wh = Wh.view(B, N, self.n_heads, self.head_dim)           # (B, N, n_heads, head_dim)
        Wh = Wh.permute(0, 2, 1, 3).contiguous()                  # (B, n_heads, N, head_dim)

        
        Wh_i = Wh.unsqueeze(-2)                                  # (B, n_heads, N, 1, head_dim)
        Wh_j = Wh.unsqueeze(-3)                                  # (B, n_heads, 1, N, head_dim)
        
        a_input = torch.cat([Wh_i.repeat(1, 1, 1, N, 1), 
                             Wh_j.repeat(1, 1, N, 1, 1)], dim=-1)  # (B, n_heads, N, N, 2*head_dim)
        

        e = torch.matmul(a_input, self.a.unsqueeze(1).unsqueeze(1))  # (B, n_heads, N, N, 1)
        e = e.squeeze(-1)                                         # (B, n_heads, N, N)
        e = self.leakyrelu(e)                                     # Apply LeakyReLU

        
        adj_mask = adj_mat.unsqueeze(1).bool()                   # (B, 1, N, N) as boolean mask
        e = torch.masked_fill(e, ~adj_mask, float('-inf'))       
        
        # 4. Softmax normalization
        attention = F.softmax(e, dim=-1)                         # (B, n_heads, N, N)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # 5. Neighborhood aggregation: weighted sum of neighbors' features
        h_prime = torch.matmul(attention, Wh)                    # (B, n_heads, N, head_dim)

        # 6. Apply the final combination per head and activation
        if self.concat:
            # Concatenate multi-head results
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous()   # (B, N, n_heads, head_dim)
            out = h_prime.view(B, N, self.n_heads * self.head_dim)  # (B, N, out_features)
            out = F.elu(out)  
        else:
            # Average the head outputs (for final layer)
            out = h_prime.mean(dim=1)  # (B, N, head_dim)

        # Remove batch dimension if input was unbatched
        return out.squeeze(0) if squeeze_output else out


class GATModel(nn.Module):
    def __init__(self, num_timepoints=2500, num_classes=4):
        super(GATModel, self).__init__()

        self.agacn1 = GraphAttentionLayer(in_features=num_timepoints, out_features=130, n_heads=10, concat=True, dropout=0.6)
        self.agacn2 = GraphAttentionLayer(in_features=125, out_features=70, n_heads=10, concat=True, dropout=0.6)
        self.agacn3 = GraphAttentionLayer(in_features=70, out_features=130, n_heads=10, concat=False, dropout=0.6)

        self.fc = nn.Linear(70 * 130, num_classes)

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n🔥 Total Trainable Parameters: {total_params:,}\n")
        print("🔍 Layer-wise Parameter Breakdown:")
        print("=" * 40)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name:<30} {param.numel():,} parameters")
        print("=" * 40)

    def forward(self, feature_matrix, adjacency_matrix):
        out1 = self.agacn1(feature_matrix, adjacency_matrix)
        out2 = self.agacn2(out1, adjacency_matrix)
        out3 = self.agacn3(out2, adjacency_matrix)

        cfa_out = torch.matmul(out2.transpose(1, 2), out3)
        cfa_out = cfa_out.view(cfa_out.shape[0], -1)
        out = self.fc(cfa_out)
        return out


if __name__ == '__main__':

    model = GATModel(num_timepoints=2000, num_classes=9)
    model.count_parameters()

