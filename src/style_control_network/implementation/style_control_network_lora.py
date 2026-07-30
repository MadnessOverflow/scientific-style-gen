import math
import os
from typing_extensions import Self

from unsloth import FastLanguageModel
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import AutoModel, AutoTokenizer


class MLPResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_p: float = 0.05):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-05, elementwise_affine=True),
            nn.Linear(dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=dropout_p, inplace=False)
        )

    def forward(self, x):
        return x + self.mlp(x)


class LoRAHyperNetwork(nn.Module):
    def __init__(
        self,
        lora_dim_in: int = 2560,  # hidden-size of Qwen residual stream
        lora_dim_q_out: int = 4096,  # output dimension of q_proj
        lora_dim_v_out: int = 1024,  # output dimension of v_proj
        lora_rank: int = 8,
        target_layers: List[int] = [18, 19, 20],
        embed_dim: int = 128,  # (based on Matryoshka Embedding)
        hidden_dim: int = 512,
        embedder_name: str = "Qwen/Qwen3-Embedding-0.6B",
        # Pfad zu vortrainierten MLP-Gewichten (ohne Embedder)
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.lora_dim_in = lora_dim_in
        self.lora_dim_q_out = lora_dim_q_out
        self.lora_dim_v_out = lora_dim_v_out
        self.lora_rank = lora_rank
        self.target_layers = target_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        if embedder_name == "AIDA-UPM/star":
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
            self.embedder = AutoModel.from_pretrained(
                "AIDA-UPM/star").to(torch.device("cuda"))

            self.embedder.eval()

            star_proj = nn.Sequential(
                nn.Linear(1024, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            )
        else:
            self.embedder, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=embedder_name,
                max_seq_length=8192,
                # load_in_4bit = True,      # Für fairen Vergleich zum 4B Modell (oder False für FP16)
                trust_remote_code=True,
                device_map="auto",      # Verteilt es korrekt auf der Titan
            )

            FastLanguageModel.for_inference(self.embedder)

        # Embedder einfrieren, da wir nur das Hypernet trainieren wollen
        for param in self.embedder.parameters():
            param.requires_grad = False

        self.layer_to_idx = {layer: idx for idx,
                             layer in enumerate(self.target_layers)}
        self.style_control_network_mlp = nn.ModuleDict()

        if embedder_name == "AIDA-UPM/star":
            self.style_control_network_mlp.add_module('star_proj', star_proj)

        # Initialisiere die Architektur der Kindklasse
        self._init_architecture()

        # Custom Init aufrufen (oder Gewichte aus Checkpoint laden)
        if pretrained_path is not None:
            self.load(pretrained_path)
        else:
            self._init_weights()

    def _init_architecture(self):
        raise NotImplementedError(
            "Muss von der Subklasse implementiert werden.")

    def _init_weights(self):
        raise NotImplementedError(
            "Muss von der Subklasse implementiert werden.")

    def forward(self, author_abstracts: List[List[str]]) -> Dict[int, Dict[str, torch.Tensor]]:
        raise NotImplementedError(
            "Muss von der Subklasse implementiert werden.")

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _get_qwen_embeddings(self, texts: List[str]) -> torch.Tensor:
        device = self.embedder.model.embed_tokens.weight.device
        import torch.nn.functional as F

        with torch.no_grad():  # Embedder bleibt gefroren
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors='pt'
            ).to(device)

            outputs = self.embedder.model(**inputs, use_cache=False)

            # Qwen3 verwendet das letzte Token anstelle von Mean-Pooling
            embeddings = self._last_token_pool(
                outputs.last_hidden_state, inputs['attention_mask'])

            # 1. Normalisierung (vor Truncation)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Matryoshka Truncation
            embeddings = embeddings[:, :self.embed_dim]

            # 2. Normalisierung (nach Truncation empfohlen bei Matryoshka)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _get_star_embeddings(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(torch.device("cuda"))

        style_embeddings = self.embedder(
            inputs.input_ids, attention_mask=inputs.attention_mask).pooler_output

        projected_embeddings = self.style_control_network_mlp['star_proj'](style_embeddings)

        return projected_embeddings

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        # Speichere explizit nur das Hypernetwork
        torch.save(self.style_control_network_mlp.state_dict(),
                   os.path.join(output_dir, "style_control_network.pt"))

    def load(self, path: str):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        try:
            self.style_control_network_mlp.load_state_dict(state_dict)
            self.style_control_network_mlp.to(torch.device("cuda"))
        except RuntimeError:
            print("Laden des state_dicts fehlgeschlagen.")
            raise

    def eval(self) -> Self:
        self.style_control_network_mlp.eval()
        return super().eval()


class LoRAHyperNetworkS(LoRAHyperNetwork):
    def _init_architecture(self):
        layer_depth_encoder = nn.Sequential(
            nn.Embedding(len(self.target_layers), 32),
            nn.LayerNorm(32, eps=1e-05, elementwise_affine=True)
        )

        module_emb = nn.Sequential(
            nn.Embedding(2, 32),
            nn.LayerNorm(32, eps=1e-05, elementwise_affine=True)
        )

        combined_dim = self.embed_dim + 32 + 32

        ab_emb = nn.Sequential(
            # 0: q_proj_A, 1: q_proj_B, 2: v_proj_A, 3: v_proj_B
            nn.Embedding(4, combined_dim),
            nn.LayerNorm(combined_dim, eps=1e-05, elementwise_affine=True)
        )

        rank_emb = nn.Sequential(
            nn.Embedding(self.lora_rank, combined_dim),
            nn.LayerNorm(combined_dim, eps=1e-05, elementwise_affine=True)
        )

        mixer = nn.Sequential(
            nn.Linear(in_features=combined_dim,
                      out_features=self.hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=0.05, inplace=False),
            nn.Linear(in_features=self.hidden_dim,
                      out_features=combined_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=0.05, inplace=False)
        )

        mlp1 = MLPResidualBlock(combined_dim, self.hidden_dim, dropout_p=0.05)
        mlp2 = MLPResidualBlock(combined_dim, self.hidden_dim, dropout_p=0.05)
        mlp3 = nn.Sequential(
            nn.LayerNorm(combined_dim, eps=1e-05, elementwise_affine=True),
            nn.Linear(combined_dim, self.hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=0.05, inplace=False),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.SiLU()
        )

        self.max_dim_q = max(self.lora_dim_in, self.lora_dim_q_out)
        self.max_dim_v = max(self.lora_dim_in, self.lora_dim_v_out)

        final_proj_q = nn.Linear(self.hidden_dim, self.max_dim_q)
        final_proj_v = nn.Linear(self.hidden_dim, self.max_dim_v)

        self.style_control_network_mlp.update({
            'layer_depth_encoder': layer_depth_encoder,
            'module_emb': module_emb,
            'ab_emb': ab_emb,
            'rank_emb': rank_emb,
            'mixer': mixer,
            'mlp1': mlp1,
            'mlp2': mlp2,
            'mlp3': mlp3,
            'final_proj_q': final_proj_q,
            'final_proj_v': final_proj_v
        })

    def _init_weights(self):
        for m in self.style_control_network_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)

        bound = 1.0 / math.sqrt(self.lora_dim_in)

        for head_name in ['final_proj_q', 'final_proj_v']:
            head = self.style_control_network_mlp[head_name]
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.uniform_(head.bias, -bound, bound)
                with torch.no_grad():
                    head.bias.div_(math.sqrt(2 * self.lora_rank))

    def forward(self, author_abstracts: List[List[str]]) -> Dict[int, Dict[str, torch.Tensor]]:
        ref_tensor = self.style_control_network_mlp['layer_depth_encoder'][0].weight
        device = ref_tensor.device
        target_dtype = ref_tensor.dtype

        batch_style_embeddings = []
        for abstracts_list in author_abstracts:
            if self.embedder.name_or_path == "AIDA-UPM/star":
                embs = self._get_star_embeddings(abstracts_list)
            else:
                embs = self._get_qwen_embeddings(abstracts_list)

            author_style = embs.mean(dim=0)
            batch_style_embeddings.append(author_style)

        latent = torch.stack(batch_style_embeddings).to(
            device=device, dtype=target_dtype)

        B = latent.size(0)  # Batch
        L = len(self.target_layers)  # Layer
        M = 2  # Module (q_proj, v_proj)
        AB = 2  # Matrix (A, B)
        R = self.lora_rank  # Rank

        latent_bc = latent.view(B, 1, 1, 1, 1, self.embed_dim).expand(
            B, L, M, AB, R, self.embed_dim)

        layer_indices = torch.tensor(
            [self.layer_to_idx[l] for l in self.target_layers], device=device, dtype=torch.long)
        layer_embs = self.style_control_network_mlp['layer_depth_encoder'](layer_indices)
        layer_embs_bc = layer_embs.view(
            1, L, 1, 1, 1, 32).expand(B, L, M, AB, R, 32)

        mod_indices = torch.tensor([0, 1], device=device, dtype=torch.long)
        mod_embs = self.style_control_network_mlp['module_emb'](mod_indices)
        mod_embs_bc = mod_embs.view(
            1, 1, M, 1, 1, 32).expand(B, L, M, AB, R, 32)

        mixed_input = torch.cat(
            [latent_bc, layer_embs_bc, mod_embs_bc], dim=-1)

        hidden = self.style_control_network_mlp['mixer'](mixed_input)
        hidden = self.style_control_network_mlp['mlp1'](hidden)

        ab_indices = torch.tensor(
            [0, 1, 2, 3], device=device, dtype=torch.long)
        ab_embs = self.style_control_network_mlp['ab_emb'](ab_indices)
        ab_embs_bc = ab_embs.view(1, 1, M, AB, 1, 128)
        hidden = hidden + ab_embs_bc

        hidden = self.style_control_network_mlp['mlp2'](hidden)

        rank_indices = torch.arange(
            self.lora_rank, device=device, dtype=torch.long)
        rank_embs = self.style_control_network_mlp['rank_emb'](rank_indices)
        rank_embs_bc = rank_embs.view(1, 1, 1, 1, R, 128)
        hidden = hidden + rank_embs_bc

        hidden = self.style_control_network_mlp['mlp3'](hidden)

        out_q = self.style_control_network_mlp['final_proj_q'](
            hidden[:, :, 0, :, :, :])  # [B, L, AB, R, max_dim_q]
        out_v = self.style_control_network_mlp['final_proj_v'](
            hidden[:, :, 1, :, :, :])  # [B, L, AB, R, max_dim_v]

        q_A_out = out_q[:, :, 0, :, :self.lora_dim_in]
        q_B_out = out_q[:, :, 1, :, :self.lora_dim_q_out].transpose(-1, -2)

        v_A_out = out_v[:, :, 0, :, :self.lora_dim_in]
        v_B_out = out_v[:, :, 1, :, :self.lora_dim_v_out].transpose(-1, -2)

        lora_dict = {}
        for i, layer_idx in enumerate(self.target_layers):
            lora_dict[layer_idx] = {
                'q_proj': {
                    'A': q_A_out[:, i, :, :],
                    'B': q_B_out[:, i, :, :]
                },
                'v_proj': {
                    'A': v_A_out[:, i, :, :],
                    'B': v_B_out[:, i, :, :]
                }
            }

        return lora_dict


class LoRAHyperNetworkL(LoRAHyperNetwork):
    def _init_architecture(self):
        layer_depth_encoder = nn.Sequential(
            nn.Embedding(len(self.target_layers), 36),
            nn.LayerNorm(36, eps=1e-05, elementwise_affine=True)
        )

        module_emb = nn.Sequential(
            nn.Embedding(2, 32),
            nn.LayerNorm(32, eps=1e-05, elementwise_affine=True)
        )

        combined_dim = self.embed_dim + 36 + 32

        mixer = nn.Sequential(
            nn.Linear(in_features=combined_dim,
                      out_features=self.hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=0.05, inplace=False),
            nn.Linear(in_features=self.hidden_dim,
                      out_features=combined_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=0.05, inplace=False)
        )

        mlp1 = MLPResidualBlock(combined_dim, self.hidden_dim, dropout_p=0.05)
        mlp2 = MLPResidualBlock(combined_dim, self.hidden_dim, dropout_p=0.05)
        mlp3 = nn.Sequential(
            nn.LayerNorm(combined_dim, eps=1e-05, elementwise_affine=True),
            nn.Linear(combined_dim, self.hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=0.05, inplace=False),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.SiLU()
        )

        # Variante L: 2 Heads pro Module (A und B getrennt, generieren vollen Rank)
        final_proj_q_A = nn.Linear(
            self.hidden_dim, self.lora_rank * self.lora_dim_in)
        final_proj_q_B = nn.Linear(
            self.hidden_dim, self.lora_rank * self.lora_dim_q_out)

        final_proj_v_A = nn.Linear(
            self.hidden_dim, self.lora_rank * self.lora_dim_in)
        final_proj_v_B = nn.Linear(
            self.hidden_dim, self.lora_rank * self.lora_dim_v_out)

        self.style_control_network_mlp.update({
            'layer_depth_encoder': layer_depth_encoder,
            'module_emb': module_emb,
            'mixer': mixer,
            'mlp1': mlp1,
            'mlp2': mlp2,
            'mlp3': mlp3,
            'final_proj_q_A': final_proj_q_A,
            'final_proj_q_B': final_proj_q_B,
            'final_proj_v_A': final_proj_v_A,
            'final_proj_v_B': final_proj_v_B
        })

    def _init_weights(self):
        for m in self.style_control_network_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)

        bound = 1.0 / math.sqrt(self.lora_dim_in)

        for head_name in ['final_proj_q_A', 'final_proj_q_B', 'final_proj_v_A', 'final_proj_v_B']:
            head = self.style_control_network_mlp[head_name]
            nn.init.zeros_(head.weight)

            if head.bias is not None:
                if head_name.endswith('_A'):
                    # A bekommt Zufallsinitialisierung (kompletter Vektor)
                    nn.init.uniform_(head.bias, -bound, bound)
                else:
                    # B bekommt saubere 0
                    nn.init.zeros_(head.bias)

    def forward(self, author_abstracts: List[List[str]]) -> Dict[int, Dict[str, torch.Tensor]]:
        ref_tensor = self.style_control_network_mlp['layer_depth_encoder'][0].weight
        device = ref_tensor.device
        target_dtype = ref_tensor.dtype

        batch_style_embeddings = []
        for abstracts_list in author_abstracts:
            if self.embedder.name_or_path == "AIDA-UPM/star":
                embs = self._get_star_embeddings(abstracts_list)
            else:
                embs = self._get_qwen_embeddings(abstracts_list)

            author_style = embs.mean(dim=0)
            batch_style_embeddings.append(author_style)

        latent = torch.stack(batch_style_embeddings).to(
            device=device, dtype=target_dtype)

        B = latent.size(0)
        L = len(self.target_layers)
        M = 2

        latent_bc = latent.view(B, 1, 1, self.embed_dim).expand(
            B, L, M, self.embed_dim)

        layer_indices = torch.tensor(
            [self.layer_to_idx[l] for l in self.target_layers], device=device, dtype=torch.long)
        layer_embs = self.style_control_network_mlp['layer_depth_encoder'](layer_indices)
        layer_embs_bc = layer_embs.view(1, L, 1, 36).expand(B, L, M, 36)

        mod_indices = torch.tensor([0, 1], device=device, dtype=torch.long)
        mod_embs = self.style_control_network_mlp['module_emb'](mod_indices)
        mod_embs_bc = mod_embs.view(1, 1, M, 32).expand(B, L, M, 32)

        mixed_input = torch.cat(
            [latent_bc, layer_embs_bc, mod_embs_bc], dim=-1)

        hidden = self.style_control_network_mlp['mixer'](mixed_input)
        hidden = self.style_control_network_mlp['mlp1'](hidden)
        hidden = self.style_control_network_mlp['mlp2'](hidden)
        hidden = self.style_control_network_mlp['mlp3'](hidden)  # [B, L, M, hidden_dim]

        q_A_flat = self.style_control_network_mlp['final_proj_q_A'](
            hidden[:, :, 0, :])  # [B, L, R * lora_dim_in]
        q_B_flat = self.style_control_network_mlp['final_proj_q_B'](
            hidden[:, :, 0, :])  # [B, L, R * lora_dim_q_out]

        v_A_flat = self.style_control_network_mlp['final_proj_v_A'](
            hidden[:, :, 1, :])  # [B, L, R * lora_dim_in]
        v_B_flat = self.style_control_network_mlp['final_proj_v_B'](
            hidden[:, :, 1, :])  # [B, L, R * lora_dim_v_out]

        # Reshape & Transpose für PEFT Format
        q_A_out = q_A_flat.view(B, L, self.lora_rank, self.lora_dim_in)
        q_B_out = q_B_flat.view(B, L, self.lora_rank,
                                self.lora_dim_q_out).transpose(-1, -2)

        v_A_out = v_A_flat.view(B, L, self.lora_rank, self.lora_dim_in)
        v_B_out = v_B_flat.view(B, L, self.lora_rank,
                                self.lora_dim_v_out).transpose(-1, -2)

        lora_dict = {}
        for i, layer_idx in enumerate(self.target_layers):
            lora_dict[layer_idx] = {
                'q_proj': {
                    'A': q_A_out[:, i, :, :],
                    'B': q_B_out[:, i, :, :]
                },
                'v_proj': {
                    'A': v_A_out[:, i, :, :],
                    'B': v_B_out[:, i, :, :]
                }
            }

        return lora_dict
