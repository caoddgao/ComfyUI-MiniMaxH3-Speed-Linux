"""Middle-block residual cache for MiniMax H3.

Every step runs the first and last transformer block. The relative change of the
first block's residual decides whether the middle blocks (1 .. N-2) run, or are
replaced by the middle residual cached on the last full step.
"""

import logging

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model

KEY = "te_speed_minimax_h3"
SAMPLE_STRIDE = 16          # token rows sampled for the skip signal
CPU_CHUNK_ROWS = 16384      # rows per host->device copy when the cache lives in RAM
SHORT_TRAJECTORY_STEPS = 10
REFERENCE_STEPS = 20


class _Branch:
    def __init__(self):
        self.prev_signal = None
        self.accumulated = 0.0
        self.skips_in_row = 0
        self.skip = False
        self.signal_in = None
        self.residual = None    # middle residual from the last full step
        self.snapshot = None    # block 1 input while a full step runs with the cache in RAM


class _Run:
    """State of one sampling run, dropped when the run ends."""

    def __init__(self, settings, sigmas):
        self.threshold, self.start_percent, self.end_percent, self.mcs, self.device, self.mode = settings
        self.sigmas = [float(s) for s in sigmas]
        self.steps = max(len(self.sigmas) - 1, 1)
        self.short = self.mode != "standard" and self.steps <= SHORT_TRAJECTORY_STEPS
        # few-step samplers move further per step: measure change per 1/20 of the trajectory
        self.signal_scale = self.steps / REFERENCE_STEPS if self.short else 1.0
        self.max_skips = 1 if self.short else self.mcs
        self.branches = {}
        self.full = 0
        self.skipped = 0

    def branch(self, transformer_options):
        return self.branches.setdefault(tuple(transformer_options["uuids"]), _Branch())

    def step_index(self, transformer_options):
        sigma = float(transformer_options["sigmas"][0])
        return min(range(len(self.sigmas)), key=lambda i: abs(self.sigmas[i] - sigma))

    def step_allowed(self, step):
        if self.short and not 1 <= step <= self.steps - 2:
            return False
        return self.start_percent <= step / self.steps <= self.end_percent

    def decide(self, br, signal, h, step):
        change = None
        if br.prev_signal is not None:
            prev = br.prev_signal.float()
            change = float((signal.float() - prev).abs().mean() / prev.abs().mean().clamp(min=1e-6)) * self.signal_scale
        with comfy.model_prefetch.pause_malloc_graph():
            if br.prev_signal is None or br.prev_signal.shape != signal.shape:
                br.prev_signal = torch.empty_like(signal)
            br.prev_signal.copy_(signal)

        br.skip = (change is not None
                   and br.residual is not None and br.residual.shape == h.shape
                   and br.skips_in_row < self.max_skips
                   and self.step_allowed(step)
                   and br.accumulated + change < self.threshold)
        if br.skip:
            br.accumulated += change
            br.skips_in_row += 1
            self.skipped += 1
        else:
            br.accumulated = 0.0
            br.skips_in_row = 0
            self.full += 1
        logging.debug(f"TE-Speed-MiniMaxH3: step={step} change={change} accumulated={br.accumulated:.4f} skip={br.skip}")

    def cache_device(self, h):
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "gpu":
            return h.device
        size = h.numel() * h.element_size()
        if comfy.model_management.get_free_memory(h.device) > size * 3:
            return h.device
        return torch.device("cpu")

    def begin_middle(self, br, h):
        with comfy.model_prefetch.pause_malloc_graph():
            if br.residual is None or br.residual.shape != h.shape:
                br.residual = None
                device = self.cache_device(h)
                pin = device.type == "cpu" and comfy.model_management.is_device_cuda(h.device)
                br.residual = torch.empty(h.shape, dtype=h.dtype, device=device, pin_memory=pin)
            if br.residual.device == h.device:
                br.residual.copy_(h)
            else:
                br.snapshot = h.clone()

    def end_middle(self, br, h):
        if br.snapshot is None:
            br.residual.neg_().add_(h)
        else:
            br.residual.copy_(br.snapshot.neg_().add_(h))
            br.snapshot = None

    def apply_middle(self, br, h):
        if br.residual.device == h.device:
            return h.add_(br.residual)
        for a in range(0, h.shape[0], CPU_CHUNK_ROWS):
            h[a:a + CPU_CHUNK_ROWS].add_(br.residual[a:a + CPU_CHUNK_ROWS].to(h.device, non_blocking=True))
        return h


class _BlockPatch:
    def __init__(self, block_index, last_index, previous):
        self.block_index = block_index
        self.last_index = last_index
        self.previous = previous

    def run(self, args, extra_args):
        if self.previous is None:
            return extra_args["original_block"](args)
        return self.previous(args, extra_args)

    def __call__(self, args, extra_args):
        transformer_options = args["transformer_options"]
        run = transformer_options[KEY]
        br = run.branch(transformer_options)
        h = args["img"]

        if self.block_index == 0:
            # blocks update the stream in place, keep the sampled input rows
            with comfy.model_prefetch.pause_malloc_graph():
                br.signal_in = h[::SAMPLE_STRIDE].clone()
            out = self.run(args, extra_args)
            signal = out["img"][::SAMPLE_STRIDE] - br.signal_in
            br.signal_in = None
            run.decide(br, signal, out["img"], run.step_index(transformer_options))
            return out

        if self.block_index == self.last_index:
            if not br.skip:
                run.end_middle(br, h)
            return self.run(args, extra_args)

        if br.skip:
            if self.block_index == 1:
                return {"img": run.apply_middle(br, h)}
            return {"img": h}

        if self.block_index == 1:
            run.begin_middle(br, h)
        return self.run(args, extra_args)


def _outer_sample_wrapper(settings):
    def wrapper(executor, *args, **kwargs):
        guider = executor.class_obj
        orig_model_options = guider.model_options
        model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        # installed per run on top of every other block patch so a skipped step bypasses them all
        dit = model_options["transformer_options"].setdefault("patches_replace", {}).setdefault("dit", {})
        last_index = len(guider.model_patcher.get_model_object("diffusion_model").blocks) - 1
        for i in range(last_index + 1):
            dit[("double_block", i)] = _BlockPatch(i, last_index, dit.get(("double_block", i)))
        run = _Run(settings, args[3])
        model_options["transformer_options"][KEY] = run
        guider.model_options = model_options
        try:
            return executor(*args, **kwargs)
        finally:
            guider.model_options = orig_model_options
            total = run.full + run.skipped
            if total:
                logging.info(f"TE-Speed-MiniMaxH3: skipped middle blocks on {run.skipped}/{total} model passes (acceleration {100.0 * run.skipped / total:.1f}%)")
    return wrapper


class TESpeedMiniMaxH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "processing_control_value": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005,
                                                       "tooltip": "Skip threshold for the accumulated change; higher is faster, lower keeps more detail."}),
                "processing_percent_1": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                                                   "tooltip": "Sampling progress where skipping may start."}),
                "processing_percent_2": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                                                   "tooltip": "Sampling progress where skipping stops."}),
                "mcs": ("INT", {"default": 2, "min": 1, "max": 10,
                                "tooltip": "Max consecutive skipped steps."}),
                "device": (["auto", "gpu", "cpu"], {"default": "auto",
                                                     "tooltip": "Where the middle residual is cached. auto uses VRAM when there is room, otherwise pinned RAM."}),
                "mode": (["standard", "4-step LoRA", "8-step LoRA"], {"default": "standard",
                                                                      "tooltip": "LoRA modes use a short-trajectory strategy at 10 steps or fewer."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "TE/optimization"
    DESCRIPTION = "MiniMax H3 inference acceleration."

    def patch(self, model, processing_control_value, processing_percent_1, processing_percent_2, mcs, device, mode):
        diffusion_model = model.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("TE-Speed-MiniMaxH3 only supports MiniMax H3 models")
        if len(diffusion_model.blocks) < 3:
            raise ValueError("TE-Speed-MiniMaxH3 needs at least three transformer blocks")
        settings = (processing_control_value, processing_percent_1, processing_percent_2, mcs, device, mode)
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY, _outer_sample_wrapper(settings))
        return (m,)
