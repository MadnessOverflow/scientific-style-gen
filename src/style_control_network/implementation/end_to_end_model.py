from typing_extensions import Self

import torch
import torch.nn as nn
from typing import List, Optional

from src.style_control_network.implementation.style_control_network_lin_trans import LinTransHyperNetwork
from src.utils.llm import get_qwen_model
from src.style_control_network.implementation.style_control_network_act_steer import StyleHyperNetwork
from src.style_control_network.implementation.style_control_network_lora import LoRAHyperNetworkL, LoRAHyperNetworkS
from unsloth.chat_templates import get_chat_template
import torch.nn.functional as F


def get_lora_hook(model_instance, layer_idx: int, module_name: str, lora_alpha: float, rank: int, dropout_p: float = 0.0):
    """
    Erzeugt eine permanente Hook-Funktion. Sie liest die aktuellen LoRA Gewichte 
    aus model_instance.current_steering_dict dynamisch aus.
    """
    scaling = lora_alpha / rank

    def hook(module, inputs, output):
        if model_instance.current_steering_dict is None:
            return output

        # Tuple Safety Check
        if isinstance(output, tuple):
            model_out = output[0]
        else:
            model_out = output

        # Input in das Linear-Layer (Shape: [Batch, SeqLen, In_Dim])
        x = inputs[0]

        lora_dict = model_instance.current_steering_dict[layer_idx]
        lora_A = lora_dict[module_name]['A']
        lora_B = lora_dict[module_name]['B']

        x_calc = x.to(dtype=torch.float32)
        A_calc = lora_A.to(device=x.device, dtype=torch.float32)
        B_calc = lora_B.to(device=x.device, dtype=torch.float32)

        if dropout_p > 0.0:
            x_calc = F.dropout(x_calc, p=dropout_p, training=module.training)

        lora_hidden = torch.matmul(x_calc, A_calc.transpose(-1, -2))
        lora_out = torch.matmul(lora_hidden, B_calc.transpose(-1, -2))

        new_output = model_out + (lora_out * scaling).to(model_out.dtype)

        if isinstance(output, tuple):
            return (new_output,) + output[1:]
        else:
            return new_output

    return hook


def get_activation_steering_hook(model_instance, layer_idx: int, alpha: float):
    def hook(module, inputs, output):
        if model_instance.current_steering_dict is None:
            return output

        if isinstance(output, tuple):
            model_out = output[0]
        else:
            model_out = output

        vector_for_layer = model_instance.current_steering_dict[layer_idx]
        vector = vector_for_layer.to(
            device=model_out.device, dtype=model_out.dtype)

        # Wenn der Vector 2D ist [Batch, Hidden], machen wir ihn zu 3D [Batch, 1, Hidden]
        if vector.dim() == 2:
            vector = vector.unsqueeze(1)
        elif vector.dim() == 1:
            # Falls du mal ohne Batch-Dimension reinkommst [Hidden] -> [1, 1, Hidden]
            vector = vector.unsqueeze(0).unsqueeze(0)

        new_output = model_out + vector * alpha

        if isinstance(output, tuple):
            return (new_output,) + output[1:]
        else:
            return new_output
    return hook


def get_lin_trans_hook(model_instance, layer_idx: int, alpha: float):
    """
    Hook für die Lineare Transformation (Scale & Shift).
    Entpackt das Tuple (scale_dict, shift_dict) aus current_steering_dict 
    und wendet die Formel: x * (1 + alpha * scale) + (alpha * shift) an.
    """
    def hook(module, inputs, output):
        if model_instance.current_steering_dict is None:
            return output

        if isinstance(output, tuple):
            model_out = output[0]
        else:
            model_out = output

        # Entpacken der Dictionaries
        scale_dict, shift_dict = model_instance.current_steering_dict

        scale_vector = scale_dict[layer_idx].to(
            device=model_out.device, dtype=model_out.dtype)
        shift_vector = shift_dict[layer_idx].to(
            device=model_out.device, dtype=model_out.dtype)

        # Broadcast auf [Batch, 1, Hidden], falls nötig
        if scale_vector.dim() == 2:
            scale_vector = scale_vector.unsqueeze(1)
            shift_vector = shift_vector.unsqueeze(1)
        elif scale_vector.dim() == 1:
            scale_vector = scale_vector.unsqueeze(0).unsqueeze(0)
            shift_vector = shift_vector.unsqueeze(0).unsqueeze(0)

        # Lineare Transformation mit Alpha-Skalierung und 1+Scale Ansatz
        new_output = model_out * \
            (1.0 + scale_vector * alpha) + (shift_vector * alpha)

        if isinstance(output, tuple):
            return (new_output,) + output[1:]
        else:
            return new_output
    return hook


class EndToEndSteeredLLM(nn.Module):
    def __init__(
        self,
        target_layers: List[int] = [18, 19, 20],
        alpha: float = 1.0,
        neftune_alpha: float = 5.0,
        steering_method: str = "lora",  # "lora", "activation_steering" oder "lin_trans"
        style_control_network_kwargs: Optional[dict] = None,
        pretrained_style_control_network_path: Optional[str] = None
    ):
        super().__init__()
        self.target_layers = target_layers
        self.alpha = alpha
        self.steering_method = steering_method

        self.lora_alpha = 16

        # 1. Basis-LLM (Qwen) laden und einfrieren
        self.llm, self.tokenizer = get_qwen_model()
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen3-instruct",
        )

        # Wir frieren das LLM komplett ein, da nur das Hypernetwork trainiert werden soll
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        print(f"[EndToEndSteeredLLM] Base model loaded and frozen:")
        print(self.llm)
        print("\n")

        if neftune_alpha > 0:
            embeddings = self.llm.get_input_embeddings()
            embeddings.neftune_noise_alpha = neftune_alpha
            embeddings.register_forward_hook(self._neftune_hook)

        sample_layer = self.llm.model.layers[self.target_layers[0]
                                             ] if self.target_layers else self.llm.model.layers[0]
        # In-Dim (hidden_size)
        lora_dim_in = sample_layer.self_attn.q_proj.weight.shape[1]
        # Out-Dim für Q
        lora_dim_q_out = sample_layer.self_attn.q_proj.weight.shape[0]
        # Out-Dim für V
        lora_dim_v_out = sample_layer.self_attn.v_proj.weight.shape[0]

        print(
            f"[EndToEndSteeredLLM] Dynamically extracted dims: In={lora_dim_in}, Q_out={lora_dim_q_out}, V_out={lora_dim_v_out}")

        # 2. Hypernetwork initialisieren
        if style_control_network_kwargs is None:
            style_control_network_kwargs = {}

        if self.steering_method.startswith("lora"):
            # XFormers backwards pass not working with Titan RTX (Turing architecture)
            # "NotImplementedError: No operator found for `memory_efficient_attention_backward`"
            try:
                import unsloth.utils.attention_dispatch as ad
                ad.HAS_XFORMERS = False
            except ImportError:
                pass

            if self.steering_method == "lora_s":
                print(
                    f"[EndToEndSteeredLLM] Initializing LoRA Hypernetwork small variant. Target layers: {self.target_layers}")
                self.style_control_network = LoRAHyperNetworkS(
                    lora_dim_in=lora_dim_in,
                    lora_dim_q_out=lora_dim_q_out,
                    lora_dim_v_out=lora_dim_v_out,
                    target_layers=target_layers,
                    pretrained_path=pretrained_style_control_network_path,
                    **style_control_network_kwargs
                )
            elif self.steering_method == "lora_l":
                print(
                    f"[EndToEndSteeredLLM] Initializing LoRA Hypernetwork large variant. Target layers: {self.target_layers}")
                self.style_control_network = LoRAHyperNetworkL(
                    lora_dim_in=lora_dim_in,
                    lora_dim_q_out=lora_dim_q_out,
                    lora_dim_v_out=lora_dim_v_out,
                    target_layers=target_layers,
                    pretrained_path=pretrained_style_control_network_path,
                    **style_control_network_kwargs
                )
            else:
                print(
                    f"[EndToEndSteeredLLM] Steering method '{self.steering_method}' not recognized as a LoRA variant. Defaulting to 'lora_l'. Target layers: {self.target_layers}")
                self.style_control_network = LoRAHyperNetworkL(
                    lora_dim_in=lora_dim_in,
                    lora_dim_q_out=lora_dim_q_out,
                    lora_dim_v_out=lora_dim_v_out,
                    target_layers=target_layers,
                    pretrained_path=pretrained_style_control_network_path,
                    **style_control_network_kwargs
                )

        elif self.steering_method == "activation_steering":
            print(
                f"[EndToEndSteeredLLM] Initializing Activation Steering Hypernetwork. Target layers: {self.target_layers}")
            self.style_control_network = StyleHyperNetwork(
                steering_dim=lora_dim_in,
                target_layers=target_layers,
                pretrained_path=pretrained_style_control_network_path,
                **style_control_network_kwargs
            )

        elif self.steering_method == "lin_trans":
            print(
                f"[EndToEndSteeredLLM] Initializing Linear Transformation Hypernetwork. Target layers: {self.target_layers}")
            self.style_control_network = LinTransHyperNetwork(
                steering_dim=lora_dim_in,  # FiLM / Lin-Trans greift auf dem Residual Stream
                target_layers=target_layers,
                pretrained_path=pretrained_style_control_network_path,
                **style_control_network_kwargs
            )

        else:
            raise ValueError(
                f"Unbekannte steering_method: {self.steering_method}. Nutze 'lora', 'activation_steering' oder 'lin_trans'.")

        print(
            f"[EndToEndSteeredLLM] Hypernetwork MLP:\n{self.style_control_network.style_control_network_mlp}")
        print(
            f"[EndToEndSteeredLLM] Other hyperparameters: alpha={alpha}, neftune_alpha={neftune_alpha}, style_control_network_kwargs={style_control_network_kwargs}")

        self.current_steering_dict = None
        self._register_permanent_hooks()

    def _register_permanent_hooks(self):
        """
        Registriert die Hooks dauerhaft am LLM, um Probleme mit Gradient Checkpointing 
        (Tensor Mismatches während der Recomputation) zu vermeiden.
        """
        if self.steering_method.startswith("lora"):
            for layer_idx in self.target_layers:
                target_layer = self.llm.model.layers[layer_idx]

                q_hook = get_lora_hook(
                    self, layer_idx, 'q_proj', lora_alpha=self.lora_alpha, rank=self.style_control_network.lora_rank)
                target_layer.self_attn.q_proj.register_forward_hook(q_hook)

                v_hook = get_lora_hook(
                    self, layer_idx, 'v_proj', lora_alpha=self.lora_alpha, rank=self.style_control_network.lora_rank)
                target_layer.self_attn.v_proj.register_forward_hook(v_hook)

        elif self.steering_method == "activation_steering":
            for layer_idx in self.target_layers:
                target_layer = self.llm.model.layers[layer_idx]
                hook = get_activation_steering_hook(
                    self, layer_idx, alpha=self.alpha)
                target_layer.register_forward_hook(hook)

        elif self.steering_method == "lin_trans":
            for layer_idx in self.target_layers:
                target_layer = self.llm.model.layers[layer_idx]
                hook = get_lin_trans_hook(self, layer_idx, alpha=self.alpha)
                target_layer.register_forward_hook(hook)

    def _neftune_hook(self, module, inputs, output):
        if module.training:
            dims = torch.tensor(output.size(1) * output.size(2))
            mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
            output = output + \
                torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
        return output

    def save(self, output_dir: str):
        self.style_control_network.save(output_dir)

    def load(self, path: str):
        self.style_control_network.load(path)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            if hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()

            self.llm.config.use_cache = False
            print(
                "[EndToEndSteeredLLM] Gradient Checkpointing für LLM aktiviert und use_cache deaktiviert!")
        else:
            print("Warnung: Das Basis-LLM unterstützt kein Gradient Checkpointing.")

    def eval(self) -> Self:
        self.style_control_network.eval()
        return super().eval()

    @property
    def config(self):
        return self.llm.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        author_abstracts: List[List[str]],
        labels: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[int] = None,
        **kwargs
    ):
        self.current_steering_dict = self.style_control_network(author_abstracts)
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            num_items_in_batch=num_items_in_batch,
            **kwargs
        )
        # WICHTIG: current_steering_dict darf hier NICHT auf None gesetzt werden!
        # Der Backward-Pass (und damit das Recomputing des Gradient Checkpointings)
        # passiert erst NACHDEM diese Funktion zurückkehrt. PyTorch braucht das Dict noch!
        return outputs

    @torch.inference_mode()
    def generate_styled_abstract(self, prompt: str, author_abstracts: List[str], max_new_tokens: int = 1024, **gen_kwargs):
        self.current_steering_dict = self.style_control_network([author_abstracts])
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False,
                                truncation=True, max_length=4096).to(self.llm.device)
        return self._run_generation(inputs, max_new_tokens, **gen_kwargs)

    def _run_generation(self, inputs, max_new_tokens, **gen_kwargs):
        generation_params = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
            "repetition_penalty": 1.05
        }
        generation_params.update(gen_kwargs)

        try:
            outputs = self.llm.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_params
            )
        finally:
            self.current_steering_dict = None

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        styled_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)
        return styled_text
