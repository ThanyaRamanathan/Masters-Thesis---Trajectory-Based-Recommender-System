import torch
import torch.nn as nn

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Detect device capabilities
HAS_CUDA = torch.cuda.is_available()


class TextSummariser(nn.Module):
    def __init__(
        self,
        base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        lora_model_name: str = "tloen/alpaca-lora-7b",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_summary_length: int = 128,
        use_lora: bool = True,
    ):
        super().__init__()
        self.max_summary_length = max_summary_length
        self.embedding_dim = embedding_dim

        self.use_lora = use_lora and PEFT_AVAILABLE

        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                padding_side="left",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Warning: could not load tokenizer for {base_model_name}: {e}. Using fallback.")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                padding_side="left",
                trust_remote_code=True,
            )

        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        try:
            if self.use_lora:
                # On GPU with sufficient memory, use 8-bit quantization
                # On CPU, use regular float32 or float16 for efficiency
                if HAS_CUDA:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        trust_remote_code=True,
                        device_map="auto",
                        load_in_8bit=True,
                    )
                else:
                    # CPU: use regular loading, optionally with torch.float16 for memory
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        trust_remote_code=True,
                        device_map="cpu",
                        dtype=torch.float32,
                    )
                self.llm_model = PeftModel.from_pretrained(
                    self.llm_model,
                    lora_model_name,
                )
            else:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    lora_model_name,
                    trust_remote_code=True,
                    device_map="cpu" if not HAS_CUDA else "auto",
                )
        except Exception as e:
            print(f"Warning: could not load LoRA model {lora_model_name}: {e}. Using base model only.")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                device_map="cpu" if not HAS_CUDA else "auto",
            )

        self.llm_model.eval()

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)

    def generate_summary(self, texts):
        device = next(self.llm_model.parameters()).device
        inputs = self.llm_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=self.max_summary_length,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.llm_tokenizer.pad_token_id,
            )
        summaries = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return summaries

    def forward(self, texts):
        summaries = self.generate_summary(texts)
        device = next(self.embedding_model.parameters()).device
        tokens = self.embedding_tokenizer(
            summaries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        ).to(device)
        outputs = self.embedding_model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


class GETNextFlowModel(nn.Module):
    def __init__(
        self,
        num_pois: int,
        num_categories: int,
        num_time_bins: int,
        config,
    ):
        super().__init__()
        self.poi_embed = nn.Embedding(num_pois, config.poi_embed_dim, padding_idx=0)
        self.cat_embed = nn.Embedding(num_categories, config.cat_embed_dim, padding_idx=0)
        self.time_embed = nn.Embedding(num_time_bins, config.time_embed_dim, padding_idx=0)
        self.flow_proj = nn.Linear(2, config.hidden_dim)
        self.summary_proj = nn.Linear(config.summary_embed_dim, config.hidden_dim)
        self.intent_proj = nn.Linear(config.intent_feature_dim, config.hidden_dim)
        self.input_proj = nn.Linear(config.poi_embed_dim + config.cat_embed_dim + config.time_embed_dim + config.hidden_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.poi_predict = nn.Linear(config.hidden_dim, num_pois)
        self.cat_predict = nn.Linear(config.hidden_dim, num_categories)
        self.time_predict = nn.Linear(config.hidden_dim, num_time_bins)
        self.dropout = nn.Dropout(0.1)
        self.summary_embedding = nn.Parameter(torch.randn(config.summary_embed_dim))

    def forward(self, poi_seq, cat_seq, time_seq, flow_feat, summary_embed, attention_mask, intent_feat=None):
        poi_embed = self.poi_embed(poi_seq)
        cat_embed = self.cat_embed(cat_seq)
        time_embed = self.time_embed(time_seq)
        flow_hidden = self.flow_proj(flow_feat)
        seq_embed = torch.cat([poi_embed, cat_embed, time_embed, flow_hidden], dim=-1)
        seq_embed = self.input_proj(seq_embed)
        seq_embed = self.dropout(seq_embed)

        summary_context = self.summary_proj(summary_embed).unsqueeze(1)
        seq_embed = seq_embed + summary_context
        if intent_feat is not None:
            intent_context = self.intent_proj(intent_feat).unsqueeze(1)
            seq_embed = seq_embed + intent_context

        src_key_padding_mask = attention_mask == 0
        transformer_out = self.transformer(seq_embed, src_key_padding_mask=src_key_padding_mask)

        pooled = transformer_out[:, -1, :]
        poi_logits = self.poi_predict(pooled)
        cat_logits = self.cat_predict(pooled)
        time_logits = self.time_predict(pooled)
        return poi_logits, cat_logits, time_logits


class GETNextModelWrapper(nn.Module):
    def __init__(self, model, summariser=None, use_intent=False):
        super().__init__()
        self.model = model
        self.summariser = summariser
        self.use_intent = use_intent

    def forward(self, poi_seq, cat_seq, time_seq, flow_feat, history_text, attention_mask, intent_feat=None):
        if self.summariser is None:
            batch_size = poi_seq.size(0)
            summary_embeddings = self.model.summary_embedding.unsqueeze(0).expand(batch_size, -1)
        else:
            summary_embeddings = self.summariser(history_text)
        active_intent = intent_feat if self.use_intent else None
        return self.model(poi_seq, cat_seq, time_seq, flow_feat, summary_embeddings, attention_mask, active_intent)
