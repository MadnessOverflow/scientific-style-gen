import os
from typing_extensions import Self

from unsloth import FastLanguageModel
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
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


class StyleHyperNetwork(nn.Module):
    def __init__(
        self,
        steering_dim: int = 4096,  # hidden-size of Qwen4B-Instruct-2407
        target_layers: List[int] = [18, 19, 20],
        embed_dim: int = 128,  # (based on Matryoshka Embedding)
        hidden_dim: int = 512,
        embedder_name: str = "Qwen/Qwen3-Embedding-0.6B",
        # Pfad zu vortrainierten MLP-Gewichten (ohne Embedder)
        pretrained_path: Optional[str] = None,
    ):
        """
        Ein Hypernetwork, das basierend auf Text-Abstracts Activation Steering Vektoren für ein LLM generiert.

        Args:
            steering_dim: Dimension der generierten Steuerungvektoren (entspricht hidden_size des LLMs)
            target_layers: Liste der Layer-Indizes, die (mit demselben Vektor) gesteuert werden sollen
            embed_dim: Matryoshka Output-Dimension (wird nach Embedding abgeschnitten)
            hidden_dim: Versteckte Dimension(en) für das MLP
            embedder_name: HuggingFace ID für das Qwen3 Embedding Modell
            pretrained_path: Optionaler Pfad zu gespeicherten MLP-Gewichten. Falls angegeben, werden diese
                             nach der Initialisierung geladen (Embedder bleibt immer frisch geladen).
        """
        super().__init__()
        self.steering_dim = steering_dim
        self.target_layers = target_layers
        self.embed_dim = embed_dim

        # 1. Embedder laden
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

        # 2a. Layer Depth Encoder + Mixer
        self.layer_to_idx = {layer: idx for idx,
                             layer in enumerate(self.target_layers)}

        layer_depth_encoder = nn.Sequential(
            nn.Embedding(len(target_layers), 36),
            nn.LayerNorm(36, eps=1e-05, elementwise_affine=True)
        )

        mixed_dim = embed_dim + 36  # 128 + 36 = 164
        mixer = nn.Sequential(
            nn.Linear(in_features=mixed_dim,
                      out_features=hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=0.05, inplace=False),
            nn.Linear(in_features=hidden_dim,
                      out_features=mixed_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=0.05, inplace=False)
        )

        # 2b. Hypernetwork MLPs (wir passen die Input-Dimension an)
        mlp1 = MLPResidualBlock(mixed_dim, hidden_dim, dropout_p=0.05)
        mlp2 = MLPResidualBlock(mixed_dim, hidden_dim, dropout_p=0.05)
        mlp3 = nn.Sequential(
            nn.LayerNorm(mixed_dim, eps=1e-05, elementwise_affine=True),
            nn.Linear(mixed_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=0.05, inplace=False),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU()
        )
        final_proj = nn.Linear(hidden_dim, steering_dim)

        style_control_network_modules = {
            'layer_depth_encoder': layer_depth_encoder,
            'mixer': mixer,
            'core': nn.Sequential(mlp1, mlp2, mlp3, final_proj)
        }

        if embedder_name == "AIDA-UPM/star":
            style_control_network_modules.update({'star_proj': star_proj})

        self.style_control_network_mlp = nn.ModuleDict(style_control_network_modules)

        # 3. Custom Init aufrufen (oder Gewichte aus Checkpoint laden)
        if pretrained_path is not None:
            self.load(pretrained_path)
        else:
            self._init_weights()

    def _init_weights(self):
        """
        Eigene Initialisierungs-Funktion für das Hypernetwork MLP.
        """
        for m in self.style_control_network_mlp.modules():
            if isinstance(m, nn.Linear):
                # Kaiming/He Init für ReLU/GELU basierte innerste Netzwerke
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        # Den letzten Layer mit 0 initialisieren, sodass der initiale Steering Vektor 0 ist
        nn.init.zeros_(self.style_control_network_mlp['core'][-1].weight)
        if self.style_control_network_mlp['core'][-1].bias is not None:
            nn.init.zeros_(self.style_control_network_mlp['core'][-1].bias)

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _get_qwen_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Nimmt Texte, gibt sie durch den Qwen Embedder und nutzt die Matryoshka-
        Eigenschaft, indem es die Vektoren auf embed_dim abschneidet.
        Nutzt neu die Qwen3 spezifische last_token_pool Strategie.
        """

        # format to qwen3 embedding input TODO
        # texts = [f"Instruct: Given a scientific abstract, retrieve relevant passages that best match the specific writing style\nQuery: {text}" for text in texts]

        # Der Embedder ist u.U. Pipeline-Parallel ("auto"), weshalb die Inputs
        # strikt auf dem Device seines ersten Layers landen müssen (nicht zwangsläufig dem des MLPs!)
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

    def forward(self, author_abstracts: List[List[str]]) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Forward Pass zur Berechnung der Steering Vektoren.

        Args:
            author_abstracts: Eine Liste (Batch), in der jedes Element eine Liste von Abstracts (Strings) ist.
                              Bsp: Batch-Size 2 mit je 3 Abstracts für einen Autor:
                              [[Abstr1_A, Abstr2_A, Abstr3_A], [Abstr1_B, Abstr2_B, Abstr3_B]]

        Returns:
            Ein Dictionary, das die Target-Layer auf die zugehörigen Steering Vektoren abbildet. 
            (aktuell hat jedes Layer pro Batch-Eintrag denselben Vektor).
        """
        # Explizite Referenz auf einen konkreten Tensor (viele Distributed-Frameworks
        # wie DataParallel stören das Verhalten von parameter() Iteratoren in ModuleDicts)
        ref_tensor = self.style_control_network_mlp['layer_depth_encoder'][0].weight
        device = ref_tensor.device
        target_dtype = ref_tensor.dtype

        batch_style_embeddings = []
        for abstracts_list in author_abstracts:
            # 1. Hole Embeddings für die Abstracts dieses einen Autors (Shape: [num_abstracts, 64])
            if self.embedder.name_or_path == "AIDA-UPM/star":
                embs = self._get_star_embeddings(abstracts_list)
            else:
                embs = self._get_qwen_embeddings(abstracts_list)

            # 2. Mean-Pooling über die Abstracts des Authors -> 1 Stil-Vektor
            author_style = embs.mean(dim=0)
            batch_style_embeddings.append(author_style)

        # 3. Stacken zu Batch-Tensor (Shape: [batch_size, 128]) und dtype an das MLP angleichen
        # (WICHTIG: Das MLP muss im Training FP32 bleiben (für PyTorch AMP GradScaler),
        # während der Embedder u.U. FP16/BF16 ausspuckt. Der Cast löst dies elegant.)
        latent = torch.stack(batch_style_embeddings).to(
            device=device, dtype=target_dtype)

        # 4. Generiere nun INDIVIDUELLE Steering Vektoren pro Target Layer
        steering_dict = {}
        for layer_idx in self.target_layers:
            # a) Numerischen Layer-Index zu Embedding-Index mappen und Embedding quer durch den Batch erzeugen
            emb_idx = torch.tensor(
                [self.layer_to_idx[layer_idx]], device=device, dtype=torch.long)
            layer_emb = self.style_control_network_mlp['layer_depth_encoder'](
                emb_idx)     # Shape: [1, 36]
            # Broadcast auf Batch, Shape: [batch_size, 36]
            layer_emb = layer_emb.expand(latent.size(0), -1)

            # b) Concatenation (Style + Layer-Depth) -> Shape: [batch_size, 164]
            mixed_latent = torch.cat([latent, layer_emb], dim=-1)

            # c) Mixer: Residual Connection + Feature Interaktion
            mixed_latent = mixed_latent + \
                self.style_control_network_mlp['mixer'](mixed_latent)

            # d) Ab durch das Core-MLP auf Steering-Dimension hochprojizieren
            steering_vector = self.style_control_network_mlp['core'](mixed_latent)

            # Pack in Dict
            steering_dict[layer_idx] = steering_vector

        return steering_dict
