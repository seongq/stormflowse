import argparse
from argparse import ArgumentParser
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.odes import ODERegistry
from sgmse.model import ScoreModel, DiscriminativeModel, StochasticRegenerationModel

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

def get_argparse_groups(parser):
	groups = {}
	for group in parser._action_groups:
		group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
		groups[group.title] = argparse.Namespace(**group_dict)
	return groups


if __name__ == '__main__':

	# throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
	base_parser = ArgumentParser(add_help=False)
	parser = ArgumentParser()
	for parser_ in (base_parser, parser):
		parser_.add_argument("--mode", required=True, choices=["score-only", "denoiser-only", "regen-freeze-denoiser", "regen-joint-training"],
			help="score-only calls the ScoreModel class, \
				  denoiser-only calls the DiscriminativeModel class, \
				  regen-... calls the StochasticRegenerationModel class with the following options: \
				  	- regen-freeze-denoiser will freeze the denoiser, make sure to call a pretrained model \
					- regen-joint-training will not freeze the denoiser and consequently will train jointly the denoiser and score model")
		parser_.add_argument("--backbone_denoiser", type=str, choices=["none"] + BackboneRegistry.get_all_names(), default="ncsnpp")
		parser_.add_argument("--pretrained_denoiser", default=None, help="checkpoint for denoiser")
		parser_.add_argument("--backbone_score", type=str, choices=["none"] + BackboneRegistry.get_all_names(), default="ncsnpp")
		parser_.add_argument("--pretrained_score", default=None, help="checkpoint for score")

		parser_.add_argument("--ode", type=str, choices=ODERegistry.get_all_names(), default="flowmatching")
		parser_.add_argument("--nolog", action='store_true', help="Turn off logging (for development purposes)")
		parser_.add_argument("--logstdout", action="store_true", help="Whether to print the stdout in a separate file")
		parser_.add_argument("--discriminatively", action="store_true", help="Train the backbone as a discriminative model instead")
	temp_args, _ = base_parser.parse_known_args()

	if "regen" in temp_args.mode:
		model_cls = StochasticRegenerationModel
	elif temp_args.mode == "score-only": 
		model_cls = ScoreModel
	elif temp_args.mode == "denoiser-only":
		model_cls = DiscriminativeModel

	# Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
	backbone_cls_denoiser = BackboneRegistry.get_by_name(temp_args.backbone_denoiser) if temp_args.backbone_denoiser != "none" else None
	backbone_cls_score = BackboneRegistry.get_by_name(temp_args.backbone_score) if temp_args.backbone_score != "none" else None
	ode_class = ODERegistry.get_by_name(temp_args.ode)
	parser = pl.Trainer.add_argparse_args(parser)
	model_cls.add_argparse_args(
		parser.add_argument_group(model_cls.__name__, description=model_cls.__name__))
	ode_class.add_argparse_args(
		parser.add_argument_group("ODE", description=ode_class.__name__))
			
	if temp_args.backbone_denoiser != "none":
		backbone_cls_denoiser.add_argparse_args(
			parser.add_argument_group("BackboneDenoiser", description=backbone_cls_denoiser.__name__))
	else:
		parser.add_argument_group("BackboneDenoiser", description="none")

	if temp_args.backbone_score != "none":
		backbone_cls_score.add_argparse_args(
			parser.add_argument_group("BackboneScore", description=backbone_cls_score.__name__))
	else:
		parser.add_argument_group("BackboneScore", description="none")

	# Add data module args
	data_module_cls = SpecsDataModule
	data_module_cls.add_argparse_args(
		parser.add_argument_group("DataModule", description=data_module_cls.__name__))
	# Parse args and separate into groups
	args = parser.parse_args()
	arg_groups = get_argparse_groups(parser)

	# Initialize logger, trainer, model, datamodule
	if "regen" in temp_args.mode:
		model = model_cls(
			mode=args.mode, backbone_denoiser=args.backbone_denoiser, backbone_score=args.backbone_score, ode=args.ode, data_module_cls=data_module_cls,
			**{
				**vars(arg_groups['StochasticRegenerationModel']),
				**vars(arg_groups['ODE']),
				**vars(arg_groups['BackboneDenoiser']),
				**vars(arg_groups['BackboneScore']),
				**vars(arg_groups['DataModule'])
			},
			nolog=args.nolog
		)
		if temp_args.pretrained_denoiser is not None:
			model.load_denoiser_model(temp_args.pretrained_denoiser)
		if temp_args.pretrained_score is not None:
			model.load_score_model(torch.load(temp_args.pretrained_score))
		data_tag = model.data_module.base_dir.strip().split("/")[-3] if model.data_module.format == "whamr" else model.data_module.base_dir.strip().split("/")[-1] 
		logging_name = f"mode={model.mode}_ode={ode_class.__name__}_score={temp_args.backbone_score}_denoiser={temp_args.backbone_denoiser}_condition={model.condition}_data={model.data_module.format}_ch={model.data_module.spatial_channels}"
	
	elif temp_args.mode == "score-only": 
		model = model_cls(
			backbone=args.backbone_score, ode=args.ode, data_module_cls=data_module_cls,
			**{
				**vars(arg_groups['ScoreModel']),
				**vars(arg_groups['ODE']),
				**vars(arg_groups['BackboneScore']),
				**vars(arg_groups['DataModule'])
			},
			nolog=args.nolog
		)
		data_tag = model.data_module.base_dir.strip().split("/")[-3] if model.data_module.format == "whamr" else model.data_module.base_dir.strip().split("/")[-1] 
		logging_name = f"mode=score-only_ode={ode_class.__name__}_backbone={args.backbone_score}_data={model.data_module.format}_ch={model.data_module.spatial_channels}"

	elif temp_args.mode == "denoiser-only": 
		model = model_cls(
			backbone=args.backbone_denoiser, ode=args.ode, data_module_cls=data_module_cls, discriminative=True,
			**{
				**vars(arg_groups['DiscriminativeModel']),
				**vars(arg_groups['ODE']),
				**vars(arg_groups['BackboneDenoiser']),
				**vars(arg_groups['DataModule'])
			},
			nolog=args.nolog
		)
		data_tag = model.data_module.base_dir.strip().split("/")[-3] if model.data_module.format == "whamr" else model.data_module.base_dir.strip().split("/")[-1] 
		logging_name = f"mode=denoiser-only_ode={ode_class.__name__}_backbone={args.backbone_denoiser}_data={model.data_module.format}_ch={model.data_module.spatial_channels}"

	logger = TensorBoardLogger(save_dir=f"./.logs/", name=logging_name, flush_secs=30) if not args.nolog else None

	# Callbacks
	callbacks = []
	callbacks.append(EarlyStopping(monitor="valid_loss", mode="min", patience=50))
	callbacks.append(TQDMProgressBar(refresh_rate=1))
	if not args.nolog:
		callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"), 
			save_last=True, save_top_k=5, monitor="valid_loss", filename='{epoch}'))
		callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"), 
			save_top_k=5, monitor="ValidationPESQ", mode="max", filename='{epoch}-{pesq:.2f}'))

	# Initialize the Trainer and the DataModule
	trainer = pl.Trainer.from_argparse_args(
		arg_groups['pl.Trainer'],
		strategy=DDPPlugin(find_unused_parameters=True), 
		logger=logger,
		log_every_n_steps=10, num_sanity_val_steps=1, devices=[2,3],
		callbacks=callbacks,
		max_epochs=1000, accelerator="gpu"
	)

	# Train model
	trainer.fit(model)