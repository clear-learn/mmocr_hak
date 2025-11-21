# -*- coding: utf-8 -*-
"""
Custom hooks for MMOCR training
wandb 로깅에서 train loss는 step 기준, validation metric은 epoch 기준으로 기록
"""

from mmengine.hooks import Hook
from mmocr.registry import HOOKS


@HOOKS.register_module()
class WandbMetricSeparatorHook(Hook):
    """
    wandb에서 train loss는 step으로, validation metric은 epoch으로 분리 기록하는 hook
    """

    def __init__(self, priority='VERY_LOW'):
        super().__init__()
        self.priority = priority
        self._wandb_initialized = False

    def before_train(self, runner):
        """학습 시작 전 wandb metric 정의"""
        # wandb backend 찾기
        wandb_backend = None
        if hasattr(runner, 'visualizer') and hasattr(runner.visualizer, '_vis_backends'):
            for backend in runner.visualizer._vis_backends.values():
                if 'Wandb' in type(backend).__name__:
                    wandb_backend = backend
                    break

        if wandb_backend is None:
            return

        # wandb 객체 가져오기
        try:
            import wandb
            if wandb.run is not None:
                # train 메트릭은 step 기준
                wandb.define_metric("train/*", step_metric="train/step")

                # validation 메트릭은 epoch 기준
                wandb.define_metric("val/*", step_metric="val/epoch")
                wandb.define_metric("Outdoor/*", step_metric="val/epoch")  # dataset prefix

                self._wandb_initialized = True
                runner.logger.info("[WandbMetricSeparatorHook] Initialized: train→step, val→epoch")
        except ImportError:
            runner.logger.warning("[WandbMetricSeparatorHook] wandb not installed, skipping")
        except Exception as e:
            runner.logger.warning(f"[WandbMetricSeparatorHook] Failed to initialize: {e}")

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """train iteration 후 step 기록"""
        if not self._wandb_initialized:
            return

        try:
            import wandb
            if wandb.run is not None:
                # train/step 메트릭 추가
                wandb.log({"train/step": runner.iter}, commit=False)
        except Exception:
            pass

    def after_val_epoch(self, runner, metrics=None):
        """validation epoch 후 epoch 기록"""
        if not self._wandb_initialized:
            return

        try:
            import wandb
            if wandb.run is not None:
                # val/epoch 메트릭 추가
                wandb.log({"val/epoch": runner.epoch}, commit=False)
        except Exception:
            pass