import logging
import warnings
from argparse import Namespace
from math import ceil
import sys

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from layers.embed import PatchEmbedding, Patching
from layers.revin import RevIN

from moment.common import TASKS
from moment.data.base import TimeseriesOutputs
from moment.utils.masking import Masking
from moment.utils.utils import (
    NamespaceWithDefaults,
    get_anomaly_criterion,
    get_huggingface_model_dimensions,
)



SUPPORTED_HUGGINGFACE_MODELS = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x seq_len], where seq_len = n_patches * patch_len
        """
        x = self.linear(
            self.dropout(x)
        )  # [batch_size x n_channels x n_patches x patch_len]
        x = x.flatten(start_dim=2, end_dim=3)  # [batch_size x n_patches x seq_len]
        return x


class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_channels * d_model, n_classes)

    def forward(self, x, input_mask: torch.Tensor = None):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_classes]
        """
        x = x.nanmean(
            dim=-1
        ).squeeze()  # x: batch_size x n_channels x n_patches x d_model
        x = self.flatten(x)  # x: batch_size x n_channels * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: batch_size x n_classes
        return y


class ForecastingHead(nn.Module):
    def __init__(
        self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: int = 0
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x forecast_horizon]
        """

        x = self.flatten(x)  # x: batch_size x n_channels x n_patches x d_model
        x = self.linear(x)  # x: batch_size x n_channels x n_patches*d_model
        x = self.dropout(x)  # x: batch_size x n_channels x forecast_horizon
        return x


class MOMENT(nn.Module):
    def __init__(self, configs: Namespace | dict, **kwargs: dict):
        super().__init__()
        configs = self._update_inputs(configs, **kwargs)
        configs = self._validate_inputs(configs)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len

        # Normalization, patching and embedding
        self.normalizer = RevIN(
            num_features=1, affine=configs.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=configs.patch_len, stride=configs.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            seq_len=configs.seq_len,
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=configs.getattr("dropout", 0.1),
            add_positional_embedding=configs.getattr("add_positional_embedding", True),
            value_embedding_bias=configs.getattr("value_embedding_bias", False),
            orth_gain=configs.getattr("orth_gain", 1.41),
        ).to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0))

        # Transformer backbone
        self.encoder = self._get_transformer_backbone(configs)

        # Prediction Head
        self.head = self._get_head(self.task_name)

        # Image encoder
        # (B, ch, H, W) -> (B, 1024, 2, 64)
        self.keogram_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 1), stride=1, padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
            
            nn.Conv2d(64, 128, kernel_size=(7, 1), stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),

            nn.Conv2d(128, 256, kernel_size=(7, 1), stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),

            nn.Conv2d(256, 512, kernel_size=(7, 1), stride=1, padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),

            nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=1, padding=(0, 0)),
            # nn.ReLU(),

        )


    def _update_inputs(
        self, configs: Namespace | dict, **kwargs
    ) -> NamespaceWithDefaults:
        if isinstance(configs, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**configs, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(configs)

    def _validate_inputs(self, configs: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            configs.transformer_backbone == "PatchTST"
            and configs.transformer_type != "encoder_only"
        ):
            warnings.warn("PatchTST only supports encoder-only transformer backbones.")
            configs.transformer_type = "encoder_only"
        if (
            configs.transformer_backbone != "PatchTST"
            and configs.transformer_backbone not in SUPPORTED_HUGGINGFACE_MODELS
        ):
            raise NotImplementedError(
                f"Transformer backbone {configs.transformer_backbone} not supported."
                f"Please choose from {SUPPORTED_HUGGINGFACE_MODELS} or PatchTST."
            )
        if (
            configs.d_model is None
            and configs.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            configs.d_model = get_huggingface_model_dimensions(
                configs.transformer_backbone
            )
            logging.info("Setting d_model to {}".format(configs.d_model))
        elif configs.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone \
                             unless transformer backbone is a Huggingface model."
            )

        if configs.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of ['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if configs.patch_stride_len != configs.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return configs

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name in {
            TASKS.PRETRAINING,
            TASKS.ANOMALY_DETECTION,
            TASKS.IMPUTATION,
        } or (
            task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.finetuning_mode == "zero-shot"
        ):
            return PretrainHead(
                self.configs.d_model,
                self.configs.patch_len,
                self.configs.getattr("dropout", 0.1),
                self.configs.getattr("orth_gain", 1.41),
            )
        elif task_name == TASKS.CLASSIFICATION:
            return ClassificationHead(
                self.configs.n_channels,
                self.configs.d_model,
                self.configs.num_class,
                self.configs.getattr("dropout", 0.1),
            )
        elif (task_name == TASKS.LONG_HORIZON_FORECASTING) or (
            task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.finetuning_mode != "zero-shot"
        ):
            num_patches = (
                max(self.configs.seq_len, self.configs.patch_len)
                - self.configs.patch_len
            ) // self.configs.patch_stride_len + 1
            self.head_nf = self.configs.d_model * num_patches
            return ForecastingHead(
                self.head_nf,
                self.configs.forecast_horizon,
                self.configs.getattr("head_dropout", 0.1),
            )
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, configs):
        if configs.transformer_backbone == "PatchTST":
            return self._get_patchtst_encoder(configs)
        else:
            return self._get_huggingface_transformer(configs)

    def _get_huggingface_transformer(self, configs):
        from transformers import T5Config, T5EncoderModel, T5Model

        if configs.getattr("randomly_initialize_backbone", False):
            model_config = T5Config.from_pretrained(configs.transformer_backbone)
            transformer_backbone = T5Model(model_config)
            logging.info(f"Initializing randomly initialized\
                          transformer from {configs.transformer_backbone}.")
        else:
            transformer_backbone = T5EncoderModel.from_pretrained(
                configs.transformer_backbone
            )
            logging.info(f"Initializing pre-trained \
                          transformer from {configs.transformer_backbone}.")

        if configs.transformer_type == "encoder_only":
            transformer_backbone = transformer_backbone.get_encoder()
        elif configs.transformer_type == "decoder_only":
            transformer_backbone = transformer_backbone.get_decoder()

        if configs.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def _get_patchtst_encoder(self, configs):
        from .layers.self_attention_family import AttentionLayer, FullAttention
        from .layers.transformer_encoder_decoder import Encoder, EncoderLayer

        encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            attention_dropout=configs.attention_dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        return encoder

    def embed(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask  : [batch_size x 1 x seq_len]
        """

        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )

        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                1, 1, self.configs.d_model
            )
            enc_out = (input_mask_patch_view * enc_out).sum(
                dim=1
            ) / input_mask_patch_view.sum(dim=1)
        elif reduction == "none":
            raise NotImplementedError

        return TimeseriesOutputs(
            embeddings=enc_out, input_mask=input_mask, metadata=reduction
        )

    def pretraining(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        keogram: torch.tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        batch_size, n_channels, _ = x_enc.shape
        n_channels_keogram = 2
        

        # Pass the keogram through the encoder
        # with torch.cuda.amp.autocast():
        keo_enc = self.keogram_encoder(keogram).reshape(batch_size, n_channels_keogram, 64, 1024)


        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]


        # only normalize the GHI channel
        # x_enc[:, 0, :] = self.normalizer(x=x_enc[:, 0, :].unsqueeze(1), mask=mask * input_mask, mode="norm").squeeze(1)


        # Normalization - Normalizes the input time series data
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm") #torch.Size([4, 1051, 512])
        # x_enc = self.normalizer(x=x_enc, mask=input_mask, mode='norm')
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0) #torch.Size([4, 1051, 512])
        # Some time-series are too short, so masking them out results in NaNs.

        # [batch_size x n_channels x seq_len]
        # Patches the 512 embedding into 8 pathches of size 64 (64*8=512)
        x_enc = self.tokenizer(x=x_enc) #torch.Size([4, 1051, 64, 8])
        # [batch_size x n_channels x n_patches x patch_len]

        # Patching and embedding - Upscales the path embedding size from 8 to 1024
        enc_in = self.patch_embedding(x_enc, mask=mask) #torch.Size([4, 1051, 64, 1024])

        # Can add the Image embedding here
        # GHI embedding - [4, 1, 64, 1024], Img embedding GHI embedding - [4, 1, 64, 1024]
        # => cat = [4, 1, 64, 1024]

        n_patches = enc_in.shape[2]
        n_channels += n_channels_keogram
        # Concatenate the encoded GHI with the encoded image
        enc_in = torch.cat((enc_in, keo_enc), 1)
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )


        # [batch_size * n_channels x n_patches x d_model]
        # changing the # of channels here because we add the keogram 


        


        # Encoder
        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0)
        if self.configs.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:

            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]


        # De-Normalization
        #Since we only normalized the GHI (channel 0), we only need to de-normalize the GHI
        # dec_out[:,0,:] = self.normalizer(x=dec_out[:,0,:].unsqueeze(1), mode="denorm").squeeze(1)
        # de-normalizing everything (might be better)
        dec_out = self.normalizer(x=dec_out, mode="denorm")


        # Average here
        dec_out = torch.mean(dec_out, 1).unsqueeze(1)


        illegal_output = (
            self._check_model_weights_for_illegal_values()
            if self.configs.debug
            else None
        )

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            pretrain_mask=mask,
            illegal_output=illegal_output,
        )

    def initialize_soft_prompt(self, **kwargs):
        n_soft_prompt_tokens = self.configs.n_soft_prompt_tokens
        self.soft_prompt = nn.Embedding(n_soft_prompt_tokens, self.configs.d_model)
        return self.soft_prompt

    def _cat_learned_embedding_to_input(self, prompt_embeds, enc_in) -> torch.Tensor:
        prompt_embeds = prompt_embeds.repeat(enc_in.size(0), 1, 1)
        enc_in = torch.cat([prompt_embeds, enc_in], dim=1)
        return enc_in

    def _extend_attention_mask(self, attention_mask, n_tokens):
        n_batches = attention_mask.shape[0]
        extension = torch.full((n_batches, n_tokens), 1).to(self.configs.device)
        return torch.cat([extension, attention_mask], dim=1)

    def reconstruct(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        if mask is None:
            mask = torch.ones_like(input_mask)

        batch_size, n_channels, _ = x_enc.shape
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        attention_mask = (
            Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
            .repeat_interleave(n_channels, dim=0)
            .to(x_enc.device)
        )

        n_tokens = 0
        if "prompt_embeds" in kwargs:
            prompt_embeds = kwargs["prompt_embeds"].to(x_enc.device)

            if isinstance(prompt_embeds, nn.Embedding):
                prompt_embeds = prompt_embeds.weight.data.unsqueeze(0)

            n_tokens = prompt_embeds.shape[1]

            enc_in = self._cat_learned_embedding_to_input(prompt_embeds, enc_in)

            attention_mask = self._extend_attention_mask(attention_mask, n_tokens)

        # Encoder
        if self.configs.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out[:, n_tokens:, :]

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out)

    def detect_anomalies(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        anomaly_criterion: str = "mse",
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        anomaly_criterion : str
        """
        outputs = self.reconstruct(x_enc=x_enc, input_mask=input_mask)
        self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

        anomaly_scores = self.anomaly_criterion(x_enc, outputs.reconstruction)

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=outputs.reconstruction,
            anomaly_scores=anomaly_scores,
            metadata={"anomaly_criterion": anomaly_criterion},
        )

    def long_forecast(
        self, x_enc: torch.Tensor, input_mask: torch.Tensor = None, keogram: torch.tensor = None, **kwargs
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        """


        batch_size, n_channels, _ = x_enc.shape
        n_channels_keogram = 2

        # Pass the keogram through the encoder
        # with torch.cuda.amp.autocast():
        keo_enc = self.keogram_encoder(keogram).reshape(batch_size, n_channels_keogram, 64, 1024)


        # Normalization
        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))

        n_patches = enc_in.shape[2]
        n_channels += n_channels_keogram
        # Concatenate the encoded GHI with the encoded image
        enc_in = torch.cat((enc_in, keo_enc), 1)
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )

        # Encoder
        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0)
        if self.configs.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x forecast_horizon]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        # Average here
        dec_out = torch.mean(dec_out, 1).unsqueeze(1)

        return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)

    def short_forecast(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        forecast_horizon: int = 1,
        **kwargs,
    ):
        # mask would be mask tokens which are attended to
        # and input_mask is typically unattended

        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        forecast_horizon : int
        """
        # Min-max scale input time-series, based on "Meta-learning
        # framework with applications to zero-shot time-series forecasting
        # scaler = torch.max(x_enc, dim=-1, keepdim=True)[0]
        # x_enc = x_enc / scaler

        batch_size, n_channels, seq_len = x_enc.shape
        frequency = kwargs["frequency"] if "frequency" in kwargs else None
        # NOTE: Add series decomposition

        num_masked_patches = ceil(forecast_horizon / self.patch_len)
        num_masked_timesteps = num_masked_patches * self.patch_len

        # Normalization
        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        # Shift the time-series and mask the last few timesteps for forecasting
        x_enc = torch.roll(x_enc, shifts=-num_masked_timesteps, dims=2)
        input_mask = torch.roll(input_mask, shifts=-num_masked_timesteps, dims=1)

        # Mixed results
        # Attending to mask tokens
        input_mask[:, -num_masked_timesteps:] = 1
        mask = torch.ones_like(input_mask)
        mask[:, -num_masked_timesteps:] = 0

        # Unattending to mask tokens
        # input_mask[:, -num_masked_timesteps:] = 0
        # mask = torch.ones_like(input_mask)

        # Tokenize
        x_enc = self.tokenizer(x=x_enc)

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        # Encoder
        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]

        end = -num_masked_timesteps + forecast_horizon
        end = None if end == 0 else end

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        forecast = dec_out[:, :, -num_masked_timesteps:end]

        # Rescale the forecast
        # forecast = forecast * scaler
        # dec_out = dec_out * scaler

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            forecast=forecast,
            metadata={"forecast_horizon": forecast_horizon},
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        keogram: torch.tensor = None,
        **kwargs,
    ):
        if self.task_name == TASKS.PRETRAINING:
            return self.pretraining(
                x_enc=x_enc, mask=mask, input_mask=input_mask, keogram=keogram, **kwargs
            )
        elif (
            self.task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.finetuning_mode == "zero-shot"
        ):
            return self.short_forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.LONG_HORIZON_FORECASTING or (
            self.task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.finetuning_mode != "zero-shot"
        ):
            return self.long_forecast(x_enc=x_enc, input_mask=input_mask,keogram=keogram,**kwargs)
        elif self.task_name == TASKS.ANOMALY_DETECTION:
            return self.detect_anomalies(x_enc=x_enc, input_mask=input_mask, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
        return

    def _check_model_weights_for_illegal_values(self):
        illegal_encoder_weights = (
            torch.stack([torch.isnan(p).any() for p in self.encoder.parameters()])
            .any()
            .item()
        )
        illegal_head_weights = (
            torch.stack([torch.isnan(p).any() for p in self.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_weights = (
            torch.stack(
                [torch.isnan(p).any() for p in self.patch_embedding.parameters()]
            )
            .any()
            .item()
        )

        return (
            illegal_encoder_weights
            or illegal_head_weights
            or illegal_patch_embedding_weights
        )


class MOMENTPipeline(MOMENT, PyTorchModelHubMixin):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        self.new_task = kwargs.get("model_kwargs", {}).pop("task_name", "pre-training")
        super().__init__(config, **kwargs)

    def init(self) -> None:
        if self.new_task != "pre-training":
            self.head = self._get_head(self.new_task)
